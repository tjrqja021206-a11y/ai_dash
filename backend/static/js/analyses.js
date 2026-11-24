// 분석 화면: 입력 방식 전환 → POST로 작업 등록 → 작업 ID로 상태 폴링 → 결과 표시
(() => {
  const API = "/api/v1/analyses/";
  const POLL_MS = 1500;
  const form = document.getElementById("analysis-form");
  const startBtn = document.getElementById("start-btn");
  const formError = document.getElementById("form-error");
  const stageEl = document.getElementById("stage");
  const resultEl = document.getElementById("result");
  const jobLabel = document.getElementById("job-label");
  const threshold = JSON.parse(document.getElementById("uncertain-threshold").textContent);
  let pollTimer = null;

  const csrf = () => form.querySelector("[name=csrfmiddlewaretoken]").value;
  const esc = (v) => String(v ?? "").replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const fmtTime = (iso) => (iso ? new Date(iso).toLocaleString("ko-KR") : "-");
  const STATUS_TEXT = { PENDING: "대기", STARTED: "처리 중", SUCCESS: "완료", FAILURE: "실패" };
  const LOW_KEYPOINT = 0.6;  // 업로드(전신) 영상은 0.95 이상, 몸 일부만 보이면 0.3~0.5

  // ---------- IP Webcam 실시간 미리보기 ----------
  // 0.2초마다 서버에 현재 화면을 요청 (서버가 카메라 사진을 가져와 관절을 그려 줌)
  const camImg = document.getElementById("cam-img");
  const camFrame = document.getElementById("cam-frame");
  const camStatus = document.getElementById("cam-status");
  const camToggle = document.getElementById("cam-toggle");
  const camRec = document.getElementById("cam-rec");
  const PREVIEW_MS = 200;
  const POSE_HINT = {
    ok: ["ok", "전신 인식됨 · 준비 완료"],
    no_person: ["warn", "사람이 보이지 않습니다"],
    core_missing: ["warn", "어깨·골반이 안 보입니다 · 뒤로 물러서세요"],
    sideways: ["warn", "화면이 옆으로 누워 있습니다 · 휴대폰 방향(또는 CAMERA_ROTATION)을 확인하세요"],
    legs_missing: ["warn", "발끝이 안 보입니다 · 조금 더 뒤로 가세요"],
    too_far: ["warn", "사람이 너무 작습니다 · 조금 앞으로 오세요"],
    unknown: ["", "카메라 화면 (AI 서버 응답 없음 · 관절 표시 생략)"],
  };
  let previewOn = false;
  let previewTimer = null;
  let lastBlobUrl = null;

  function setPreview(on) {
    if (!camImg) return;
    previewOn = on;
    clearTimeout(previewTimer);
    camToggle.textContent = on ? "미리보기 끄기" : "미리보기 켜기";
    if (on) { camStatus.textContent = "카메라에 연결 중…"; previewLoop(); }
    else { camStatus.textContent = "미리보기가 꺼져 있습니다."; camFrame.className = "cam-frame"; }
  }

  async function previewLoop() {
    if (!previewOn || document.hidden) return;
    const started = performance.now();
    let delay = PREVIEW_MS;
    try {
      const name = form.elements.camera_name.value;
      const res = await fetch(`${API}cameras/${encodeURIComponent(name)}/preview/`, {
        credentials: "same-origin", cache: "no-store",
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.detail || `미리보기 실패 (${res.status})`);
      }
      const url = URL.createObjectURL(await res.blob());
      camImg.src = url;
      if (lastBlobUrl) URL.revokeObjectURL(lastBlobUrl);
      lastBlobUrl = url;
      const code = res.headers.get("X-Pose-Status") || "unknown";
      const [level, text] = POSE_HINT[code] || POSE_HINT.unknown;
      const conf = res.headers.get("X-Pose-Keypoint-Conf");
      camStatus.textContent = text + (conf && code !== "no_person" ? ` (관절 ${Math.round(conf * 100)}%)` : "");
      camFrame.className = `cam-frame ${level}`;
      delay = Math.max(0, PREVIEW_MS - (performance.now() - started));
    } catch (err) {
      camStatus.textContent = err.message;
      camFrame.className = "cam-frame error";
      delay = 2000;  // 카메라가 꺼져 있으면 2초 뒤 다시 시도
    }
    if (previewOn) previewTimer = setTimeout(previewLoop, delay);
  }

  function setRecording(on) { if (camRec) camRec.hidden = !on; }

  if (camToggle) {
    camToggle.addEventListener("click", () => setPreview(!previewOn));
    // 탭을 다른 곳으로 옮겼다 돌아오면 이어서
    document.addEventListener("visibilitychange", () => { if (!document.hidden && previewOn) previewLoop(); });
  }

  // ---------- 입력 방식 탭 ----------
  document.querySelectorAll(".tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      const mode = tab.dataset.mode;
      document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t === tab));
      document.querySelectorAll(".mode").forEach((el) => { el.hidden = !el.classList.contains(`mode-${mode}`); });
      form.elements.input_type.value = mode;
      // IP Webcam 탭을 고르면 미리보기를 자동으로 켜고, 업로드 탭으로 가면 끔
      setPreview(mode === "IP_WEBCAM" && !!camImg);
    });
  });

  // ---------- 작업 등록 ----------
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    formError.hidden = true;
    const data = new FormData();
    const mode = form.elements.input_type.value;
    data.append("input_type", mode);
    data.append("title", form.elements.title.value);
    data.append("expected_action", form.elements.expected_action.value);
    if (mode === "UPLOAD") {
      const file = form.elements.video.files[0];
      if (!file) return showFormError("분석할 영상을 선택하세요.");
      data.append("video", file);
    } else {
      if (!form.elements.camera_name) return showFormError("허용된 카메라가 없습니다.");
      data.append("camera_name", form.elements.camera_name.value);
      data.append("capture_seconds", form.elements.capture_seconds.value);
    }

    startBtn.disabled = true;
    startBtn.textContent = mode === "UPLOAD" ? "업로드 중…" : "요청 중…";
    try {
      const res = await fetch(API, {
        method: "POST", body: data, credentials: "same-origin",
        headers: { "X-CSRFToken": csrf() },
      });
      const body = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(flattenErrors(body) || `요청 실패 (${res.status})`);
      form.elements.video.value = "";
      watchJob(body.id);
      loadHistory();
    } catch (err) {
      showFormError(err.message);
    } finally {
      startBtn.disabled = false;
      startBtn.textContent = "AI 분석 시작";
    }
  });

  function showFormError(msg) { formError.textContent = msg; formError.hidden = false; }
  function flattenErrors(body) {
    if (typeof body === "string") return body;
    return Object.entries(body).map(([k, v]) => `${k === "detail" || k === "non_field_errors" ? "" : k + ": "}${[].concat(v).join(" ")}`).join(" / ");
  }

  // ---------- 상태 폴링 ----------
  function setSteps(status) {
    const order = { PENDING: 0, STARTED: 1, SUCCESS: 2, FAILURE: 2 };
    document.querySelectorAll("#steps li").forEach((li, i) => {
      li.classList.toggle("done", i < order[status]);
      li.classList.toggle("current", i === order[status]);
      li.classList.toggle("fail", i === 2 && status === "FAILURE");
    });
  }

  function watchJob(id) {
    clearTimeout(pollTimer);
    jobLabel.textContent = `#${id}`;
    resultEl.innerHTML = "";
    history.replaceState(null, "", `?job=${id}`);
    const tick = async () => {
      try {
        const res = await fetch(`${API}${id}/status/`, { credentials: "same-origin" });
        if (!res.ok) throw new Error(res.status === 404 ? "작업을 찾을 수 없습니다." : `상태 조회 실패 (${res.status})`);
        const s = await res.json();
        setSteps(s.status);
        stageEl.textContent = `${STATUS_TEXT[s.status]}${s.stage ? " · " + s.stage : ""}`;
        // 서버가 카메라 영상을 받는 동안: 미리보기에 '녹화 중' 표시 (꺼져 있으면 자동으로 켬)
        const capturing = s.status === "STARTED" && (s.stage || "").startsWith("영상 수집");
        setRecording(capturing);
        if (capturing && camImg && !previewOn && form.elements.input_type.value === "IP_WEBCAM") setPreview(true);
        if (s.is_finished) { setRecording(false); await showDetail(id); loadHistory(); return; }
        pollTimer = setTimeout(tick, POLL_MS);
      } catch (err) {
        stageEl.textContent = err.message;
      }
    };
    tick();
  }

  // ---------- 결과 표시 ----------
  async function showDetail(id) {
    const res = await fetch(`${API}${id}/`, { credentials: "same-origin" });
    if (!res.ok) { resultEl.innerHTML = `<p class="alert error">결과를 불러오지 못했습니다.</p>`; return; }
    const job = await res.json();
    setSteps(job.status);
    stageEl.textContent = `${job.status_display} · ${job.input_type_display}` +
      (job.camera_name ? ` (${esc(job.camera_name)}, ${job.capture_seconds}초)` : "");

    let html = "";
    if (job.status === "FAILURE") {
      html += `<div class="alert error"><strong>분석 실패</strong><br>${esc(job.error_message)}</div>`;
    } else if (job.predictions.length === 0) {
      html += `<div class="alert warn">${esc(job.result_message || "예측 결과가 없습니다.")}</div>`;
    } else {
      html += `<p class="muted">${esc(job.result_message)}</p>`;
      // 관절 신뢰도가 낮으면 동작 신뢰도도 낮아짐 → 원인(입력 품질)과 해결 방법을 먼저 알려 줌
      const lowQuality = job.predictions.filter((p) => p.mean_keypoint_conf != null && p.mean_keypoint_conf < LOW_KEYPOINT);
      if (lowQuality.length) {
        html += `<div class="alert warn"><strong>입력 품질이 낮습니다</strong> (관절 신뢰도 ${lowQuality.map((p) => `#${p.track_id} ${Math.round(p.mean_keypoint_conf * 100)}%`).join(", ")})<br>
          몸 일부가 화면 밖이거나, 옆으로 누운 화면이거나, 흔들리면 관절을 잘 찾지 못해 동작 신뢰도도 낮아집니다.
          머리부터 발끝까지 보이게 2~3m 떨어져, 휴대폰을 고정하고 밝은 곳에서 다시 시도하세요.</div>`;
      }
      html += `<table class="table"><thead><tr><th>Track ID</th><th>예측 동작</th><th>신뢰도</th><th>후보 2·3위</th><th>창 수</th><th>보인 시간</th><th>관절 신뢰도</th></tr></thead><tbody>`;
      for (const p of job.predictions) {
        const pct = Math.round(p.confidence * 100);
        const others = p.top3.slice(1).map((t) => `${esc(t.action_label)} ${Math.round(t.probability * 100)}%`).join(", ");
        html += `<tr class="${p.is_uncertain ? "uncertain" : ""}">
          <td>#${p.track_id}</td>
          <td><strong>${esc(p.action_name)}</strong><br><span class="muted">${esc(p.action_id)} ${esc(p.action_label)}</span>
              ${p.is_uncertain ? '<br><span class="badge s-FAILURE">판단 불확실</span>' : ""}</td>
          <td><div class="bar"><span style="width:${pct}%"></span></div>${pct}%</td>
          <td class="muted">${others}</td>
          <td>${p.window_count}</td>
          <td>${p.duration_sec}초</td>
          <td>${p.mean_keypoint_conf ?? "-"}${p.mean_keypoint_conf != null && p.mean_keypoint_conf < LOW_KEYPOINT ? '<br><span class="badge s-FAILURE">품질 낮음</span>' : ""}</td></tr>`;
      }
      html += `</tbody></table>`;
    }
    const facts = [
      job.expected_action ? `수행하려던 동작: ${esc(job.expected_action)} ${esc(job.expected_action_name)}` : null,
      job.video_duration_sec != null ? `영상 ${job.video_duration_sec}초 · ${job.video_fps}fps` : null,
      job.processing_ms != null ? `AI 처리 ${(job.processing_ms / 1000).toFixed(1)}초` : null,
      job.total_seconds != null ? `요청~완료 ${job.total_seconds}초` : null,
    ].filter(Boolean);
    if (facts.length) html += `<p class="muted">${facts.join(" · ")}</p>`;
    if (job.overlay_url) {
      // Skeleton Overlay가 있으면 그것을 먼저 보여 주고, 원본은 링크로
      html += `<h3 class="overlay-title">Skeleton Overlay <span class="muted">사람별 관절·동작·신뢰도 (? = 판단 불확실)</span></h3>
        <video src="${esc(job.overlay_url)}" controls muted playsinline loop class="preview"></video>`;
      if (job.video_url) html += `<p class="muted"><a href="${esc(job.video_url)}" target="_blank" rel="noopener">원본 영상 보기</a></p>`;
    } else if (job.video_url) {
      html += `<video src="${esc(job.video_url)}" controls muted class="preview"></video>`;
    }
    resultEl.innerHTML = html;
  }

  // ---------- 최근 이력 ----------
  async function loadHistory() {
    const tbody = document.querySelector("#history tbody");
    const res = await fetch(API, { credentials: "same-origin" });
    if (!res.ok) { tbody.innerHTML = `<tr><td colspan="9">이력을 불러오지 못했습니다.</td></tr>`; return; }
    const { results } = await res.json();
    if (!results.length) { tbody.innerHTML = `<tr><td colspan="9" class="muted">아직 분석 기록이 없습니다.</td></tr>`; return; }
    tbody.innerHTML = results.map((job) => {
      const preds = job.predictions.map((p) => `#${p.track_id} ${esc(p.action_label)} ${Math.round(p.confidence * 100)}%${p.is_uncertain ? "?" : ""}`).join("<br>") || "-";
      return `<tr>
        <td>#${job.id}</td><td>${esc(job.input_type_display)}</td><td>${esc(job.title) || "-"}</td>
        <td>${job.expected_action ? esc(job.expected_action + " " + job.expected_action_name) : "-"}</td>
        <td><span class="badge s-${job.status}">${esc(job.status_display)}</span></td>
        <td>${preds}</td>
        <td>${job.processing_ms != null ? (job.processing_ms / 1000).toFixed(1) + "초" : "-"}</td>
        <td>${fmtTime(job.created_at)}</td>
        <td class="actions"><button type="button" class="link" data-view="${job.id}">상세</button>
            <button type="button" class="link danger" data-delete="${job.id}" ${job.status === "STARTED" ? "disabled" : ""}>삭제</button></td></tr>`;
    }).join("");
  }

  document.querySelector("#history").addEventListener("click", async (event) => {
    const view = event.target.dataset.view;
    const del = event.target.dataset.delete;
    if (view) { watchJob(Number(view)); window.scrollTo({ top: 0, behavior: "smooth" }); }
    if (del && confirm(`#${del} 분석 기록과 사람별 결과, 영상을 삭제할까요?`)) {
      const res = await fetch(`${API}${del}/`, { method: "DELETE", credentials: "same-origin", headers: { "X-CSRFToken": csrf() } });
      if (!res.ok) alert(flattenErrors(await res.json().catch(() => ({}))) || "삭제 실패");
      loadHistory();
    }
  });

  loadHistory();
  const initial = new URLSearchParams(location.search).get("job");
  if (initial) watchJob(Number(initial));
})();

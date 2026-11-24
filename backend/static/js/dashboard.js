// 관리자 대시보드: Django가 ORM으로 집계해 json_script로 넘겨준 값으로 Chart.js 차트를 그립니다.
(() => {
  const data = JSON.parse(document.getElementById("chart-data").textContent);
  const css = getComputedStyle(document.documentElement);
  const color = (name) => css.getPropertyValue(name).trim();
  Chart.defaults.color = color("--muted");
  Chart.defaults.borderColor = color("--line");

  new Chart(document.getElementById("action-chart"), {
    type: "bar",
    data: {
      labels: data.actions.labels,
      datasets: [{ label: "예측 건수", data: data.actions.values, backgroundColor: color("--accent"), borderRadius: 4 }],
    },
    options: {
      indexAxis: "y",
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: { x: { beginAtZero: true, ticks: { precision: 0 } } },
    },
  });

  const statusColors = { PENDING: "--s-pending", STARTED: "--s-started", SUCCESS: "--s-success", FAILURE: "--s-failure" };
  new Chart(document.getElementById("status-chart"), {
    type: "doughnut",
    data: {
      labels: data.status.labels,
      datasets: [{ data: data.status.values, backgroundColor: data.status.keys.map((k) => color(statusColors[k])), borderColor: color("--card") }],
    },
    options: { maintainAspectRatio: false, plugins: { legend: { position: "right" } } },
  });

  new Chart(document.getElementById("input-chart"), {
    type: "bar",
    data: {
      labels: data.input.labels,
      datasets: [{ label: "작업 수", data: data.input.values, backgroundColor: color("--accent-2"), borderRadius: 4 }],
    },
    options: {
      indexAxis: "y",
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: { x: { beginAtZero: true, ticks: { precision: 0 } } },
    },
  });
})();

# training — 증강 재학습·신뢰도 보정 (선택 확장)

웹 서비스의 기본 원칙은 "모델을 새로 학습하지 않는다"입니다. 이 폴더는 그 위에 더한 **선택 확장**의 코드와 결과입니다.
새 촬영 없이 기존 255개 영상(S01~S05, 5명 × 17동작 × 3방향)의 **관절 좌표만 증강해** 다시 학습했고, 모델 확률을 "정답 가능성"으로 보정했습니다.
자세한 설명과 결과는 메인 README의 [7-5](../README.md#7-5-기존-영상-증강으로-재학습)·[7-6](../README.md#7-6-신뢰도-보정--정답-가능성)에 있습니다.

## 파일

| 파일 | 내용 |
|---|---|
| `scripts/51_train_augmented_motion_model.py` | 관절 좌표 증강(fps·pad·mirror·speed·camera·noise) → 사람 교차 검증으로 비교 → 최종 모델 저장 |
| `scripts/52_fit_confidence_calibration.py` | 처음 보는 사람 예측 2,295건으로 로지스틱 회귀 보정기 학습 → `calibration.json` |
| `reports/motion_capture_aug/experiments.csv` | 증강 하나씩 효과 (원본·웹캠 15fps·웹캠 60fps 정확도) |
| `reports/motion_capture_aug/combo/`, `tune/` | 증강 조합·배수·확률 비교 → 채택: `fps+pad+speed`, ×6, p=0.7 |
| `reports/motion_capture_aug/svcpath/`, `raw48/` | 서비스 경로(59.88fps 보간) vs 원래 48번 방식 비교 |
| `reports/motion_capture_aug/calib/` | 보정 결과(`calibration.txt`: ECE 0.462 → 0.011)와 입력 데이터 `oof_predictions.csv` |
| `reports/motion_capture_aug/run_*.log` | 실행 기록 |

학습 결과 모델은 서비스가 바로 쓰는 위치에 있습니다: `ai_service/models/action_model.joblib`, `clip_model.joblib`, `model_info.json`, `calibration.json`.

## 다시 실행하기

두 스크립트 모두 Feature 계산에 **웹 서비스와 같은 코드**(`ai_service/src`, `ai_service/services`)를 import합니다. 그래서 학습과 추론의 입력이 같습니다.

```bash
# 보정기만 다시 학습 (저장소에 있는 oof_predictions.csv 사용, 몇 초)
python training/scripts/52_fit_confidence_calibration.py
# → training/artifacts/motion_capture_aug/calibration.json  (ai_service/models/calibration.json과 같은 값)
```

51번은 관절 CSV(약 72MB, 원본 학습 프로젝트 `data/motion_capture_keypoints/S01~S05`)가 필요해 저장소에는 넣지 않았습니다.
CSV를 `training/data/motion_capture_keypoints/`에 두거나 `--keypoint-root`로 경로를 지정해 실행합니다.

```bash
python training/scripts/51_train_augmented_motion_model.py --experiments all          # 증강별 비교 (오래 걸림)
python training/scripts/51_train_augmented_motion_model.py --oof <실험이름>          # 보정용 예측 수집
python training/scripts/51_train_augmented_motion_model.py --final <실험이름>        # 5명 전체로 최종 모델 저장
```

결과 모델은 `training/artifacts/`에 저장됩니다(`.gitignore`). 서비스에 적용하려면 `ai_service/models/`로 복사합니다.

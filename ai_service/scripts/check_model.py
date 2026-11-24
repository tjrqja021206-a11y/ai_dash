"""T01 모델 불러오기 검사 + feature_columns.json 생성.

실행 (ai_service 폴더에서):
    python scripts/check_model.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

AI_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AI_ROOT))

import sklearn  # noqa: E402

from services.model_loader import (  # noqa: E402
    MODEL_DIR,
    clip_feature_columns,
    load_bundle,
    window_feature_columns,
)


def main() -> None:
    bundle = load_bundle()
    summary = bundle.summary()

    print(f"scikit-learn   : {sklearn.__version__} (학습 환경 1.9.1)")
    print(f"조건           : {summary['condition']}")
    print(f"창 모델 입력   : {summary['window_features']} = 35 Frame Feature × 5 통계")
    print(f"Clip 모델 입력 : {summary['clip_features']}")
    print(f"Window         : {summary['window_sec']}초, stride {summary['stride_sec']}초, 학습 FPS {summary['train_fps']}")
    print(f"Pose 모델      : {summary['pose_model']}")
    print(f"클래스 {len(bundle.classes)}개:")
    for item in summary["classes"]:
        print(f"  {item['action_id']}  {item['action_label']:<18} {item['description']}")

    out = MODEL_DIR / "feature_columns.json"
    out.write_text(
        json.dumps(
            {
                "window_model": window_feature_columns(),
                "clip_model": clip_feature_columns(),
            },
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )
    print(f"\nFeature 순서 저장: {out}")
    print("T01 PASS")


if __name__ == "__main__":
    main()

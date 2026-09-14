# 서버 준비와 진행 상태 확인

**전체 실험은 사용자 승인 뒤 시작한다.** 서버 저장소는 `/mnt/working/mm-sae`, Python 환경은 `/mnt/working/mm-sae/.venv`, 실행 설정은 `/mnt/working/mm-sae/configs/elice.yaml`이다. 전체 데이터 학습과 RQ1 분석을 시작하기 전에 이 설정의 승인 기록이 있어야 한다. 설정 확인과 승인 요청 파일 생성은 학습을 실행하지 않는다.

```bash
cd /mnt/working/mm-sae
.venv/bin/python -m mm_sae --config configs/elice.yaml validate
.venv/bin/python -m mm_sae --config configs/elice.yaml review
```

승인 요청은 `/mnt/working/mm-sae/approvals/elice-rq1.request.json`에 저장한다. `decision`은 `pending`이다. 사용자가 실험 내용을 승인하기 전에는 `/mnt/working/mm-sae/approvals/elice-rq1.json`을 만들지 않는다. 승인 후에는 그 시점의 코드·설정·설명 문서·주요 설치 패키지 버전에 대한 승인을 기록한다. 현재 지문과 다른 승인으로는 실행할 수 없다.

**서버에서 진행 상태를 언제든 조회할 수 있다.** 다음 명령은 모델을 불러오거나 실험을 시작하지 않고 저장된 상태 파일을 읽는다. 전체 실험을 시작하지 않았으면 `not_started`를 표시한다.

```bash
/mnt/working/mm-sae/.venv/bin/python -m mm_sae \
  --run-dir /mnt/working/mm-sae/runs/elice-rq1 status
```

10초마다 다시 표시하려면 다음 명령을 사용한다.

```bash
watch -n 10 /mnt/working/mm-sae/.venv/bin/python -m mm_sae \
  --run-dir /mnt/working/mm-sae/runs/elice-rq1 status
```

기계가 읽을 값이 필요하면 명령 끝에 `--json`을 붙인다. 원본 파일은 `/mnt/working/mm-sae/runs/elice-rq1/progress.json`이다. 현재 단계와 내부 작업, 처리한 수와 총수, 경과 시간, 관측 속도와 예상 남은 시간을 저장한다. 학습 중에는 갱신 횟수와 학습 회차를 표시하고 Hugging Face가 기록한 손실·학습률도 보존한다. 디스크 여유 공간, 프로세스 최대 메모리, PyTorch의 GPU 할당량도 기록한다. GPU 할당량은 다른 프로세스를 포함한 장치 전체 사용량이 아니다.

예상 남은 시간은 현재 작업에서 관측한 처리 속도로 계산한다. 처리 비용이 다른 개념이나 오류 조합의 남은 시간은 근사치다. 시작하지 않은 작업이나 전체 반복 수가 없는 단일 행렬 계산은 미산정으로 표시한다. 모델 다운로드는 Hugging Face의 진행 출력을 로그에 남기며 상태 파일에는 모델을 불러오는 중이라고 표시한다. 전체 실행 시간은 아직 관측하지 않은 단계가 있으므로 임의로 추정하지 않는다. 단계 9개 중 3개가 끝났다고 전체 시간의 3분의 1이 끝난 것은 아니다.

같은 서버에서 상태를 조회하면 실행 프로세스가 사라졌는지도 확인한다. 상태 기록이 오래 갱신되지 않았으면 이를 표시한다. 오류로 끝나면 오류 메시지를 남기며 실패한 단계를 완료로 표시하지 않는다. 실행 자체가 승인 검사에서 중단되었다면 학습은 시작되지 않았으므로 진행 파일을 생성하지 않는다.

**승인 후에는 터미널 연결과 독립된 `tmux` 세션에서 실행한다.** 다음은 실행 예정 명령이며 준비 과정에서 실행하지 않는다. 같은 이름의 세션이 이미 있으면 먼저 그 작업을 확인하고 중복 실행하지 않는다.

```bash
mkdir -p /mnt/working/mm-sae/runs/elice-rq1
tmux new-session -d -s mm-sae-rq1 -c /mnt/working/mm-sae \
  'PYTHONUNBUFFERED=1 /mnt/working/mm-sae/.venv/bin/python -m mm_sae --config /mnt/working/mm-sae/configs/elice.yaml >> /mnt/working/mm-sae/runs/elice-rq1/console.log 2>&1'
```

전체 로그는 다음 명령으로 읽는다. `Ctrl-C`는 로그 조회를 종료하며 `tmux`의 학습을 중단하지 않는다.

```bash
tail -f /mnt/working/mm-sae/runs/elice-rq1/console.log
```

중단 뒤 같은 설정과 코드로 다시 실행하면 완료한 단계를 건너뛴다. 표현 추출은 저장한 묶음부터, 학습은 마지막 학습 상태부터 재개한다. 객체 제거 실험은 완료한 오류 조합·반복의 결과를 재사용한다. 승인한 설정이나 코드를 바꾸면 승인 기록과 결과 폴더를 새로 준비해야 한다.

**최초 서버 복제에는 Git 이력 묶음을 사용했다.** 비공개 GitHub 저장소의 인증 정보를 서버에 복사하지 않고, 로컬에서 `git bundle`로 만든 전체 이력을 서버에서 복제했다. 서버의 `origin`은 `https://github.com/JiH00nKw0n/mm-sae.git`이다. 서버에서 GitHub로 직접 가져오거나 올리려면 별도의 인증이 필요하다. 이번 준비에서 서버에 GitHub 토큰이나 개인 SSH 키를 저장하지 않았다.

서버에는 `/mnt/working/.tools/bin/uv`와 Python 3.11을 설치했다. Torch 2.6.0은 CUDA 12.4용 배포판을 먼저 설치하고, 나머지는 고정 목록으로 설치한다. 신규 환경에서 같은 방식으로 준비할 때 다음 명령을 사용한다.

```bash
cd /mnt/working/mm-sae
/mnt/working/.tools/bin/uv venv --python 3.11 .venv
/mnt/working/.tools/bin/uv pip install --python .venv/bin/python \
  torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
/mnt/working/.tools/bin/uv pip install --python .venv/bin/python -r requirements-runtime.lock
/mnt/working/.tools/bin/uv pip install --python .venv/bin/python --no-deps -e .
/mnt/working/.tools/bin/uv pip install --python .venv/bin/python \
  pytest==8.3.5 ruff==0.11.2 pyright==1.1.408
```

타입 검사는 Pyright로, 코드 오류·형식 검사는 Ruff로 수행한다. 단위 검사는 데이터 다운로드 없이 수치 정의와 재개·승인 동작을 확인한다.

```bash
cd /mnt/working/mm-sae
.venv/bin/python -m pyright --pythonpath .venv/bin/python
.venv/bin/python -m ruff check src experiments tests scripts
.venv/bin/python -m ruff format --check src experiments tests scripts
.venv/bin/python -m pytest -q
```

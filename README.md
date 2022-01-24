# 유효성 검증용 docker 사용법

- docker 이미지 import
    - docker_image 디렉토리로 이동

    - 위험기상 예측모델 docker image import
        - docker import weather_prediction.tar
    - 어노테이션 탐지모델 docker image import
        - docker import annotation.tar
    - import된 이미지의 IMAGE ID 확인
        - 추후 run에 사용되기 때문에 'docker images' 명령어를 통해 IMAGE ID를 확인해두어야함
        
    ```python
    (base) jini1114@user1:~/git$ docker import IMAGE.tar
    sha256:0e5831183a4990d2b78fd9adfc2e2dfef234bd507c5316b71aabb18306e6512b
    (base) jini1114@user1:~/git$ docker images
    REPOSITORY                       TAG       IMAGE ID       CREATED         SIZE
    <none>                           <none>    0e5831183a49   4 seconds ago   8.74GB
    ```
        
- 위험기상 예측모델 docker run
    - docker run -it --gpus all --ipc=host -e NVIDIA_VISIBLE_DEVICES=all -v /path/weather_prediction:/testdata/ --name final_test IMAGE_ID  /bin/bash
        - 여기서 -v 옵션 뒤에 있는 path는 데이터셋을 다운로드 받은 위치를 절대경로로 작성
    ```python
    (base) jini1114@user1:~/git$ docker run -it --gpus all --ipc=host -e NVIDIA_VISIBLE_DEVICES=all -v /mnt/ai-nas02/WORK/jini1114/proof_of_validity/weather_prediction:/testdata/ --name final_test 0e5831183a49  /bin/bash
    ```
- 어노테이션 탐지모델 docker run
    - docker run -it --gpus all --ipc=host -e NVIDIA_VISIBLE_DEVICES=all -v /path/annotation:/testdata/ --name final_test IMAGE_ID  /bin/bash
        - 여기서 -v 옵션 뒤에 있는 path는 데이터셋을 다운로드 받은 위치를 절대경로로 작성
    ```python
    (base) jini1114@user1:~/git$ docker run -it --gpus all --ipc=host -e NVIDIA_VISIBLE_DEVICES=all -v /mnt/ai-nas02/WORK/jini1114/proof_of_validity/weather_prediction:/testdata/ --name final_test 0e5831183a49  /bin/bash
    ```

- git 다운로드 후 쉘스크립트 실행
    - 위험기상 예측모델
    
    ```python
    git clone https://github.com/wlsdml1114/aiwx_data.git
    cd /aiwx_data/
    sh LSTM.sh
    ```
    
    - 어노테이션 탐지모델
    
    ```python
    git clone https://github.com/wlsdml1114/aiwx_data.git
    cd /aiwx_data/
    sh MASK_RCNN.sh
    ```

- 모델 결과 확인
    - test_results 디렉토리에 모델의 최종 결과를 csv형태로 저장
        - 실행 로그는 logs.txt에 저장
    - 위험기상 예측모델
        - 폭우(rain), 폭염(heat), 폭설(snow)의 이름으로 모델 예측결과인 prediction.csv와 실제 정답인 target.csv를 별도의 파일로 저장
    - 어노테이션 탐지모델
        - 16종의 어노테이션 각각에 대한 탐지결과 grid cell인 prediction.csv와 실제 정답 grid cell인 target.csv를 별도의 파일로 저장

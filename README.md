# 유효성 검증용 docker 사용법

- docker image import
    - docker import docker_image.tar
    - docker images
        - IMAGE ID 확인
        
        ```python
        (base) jini1114@user1:~/git$ docker import IMAGE.tar
        sha256:0e5831183a4990d2b78fd9adfc2e2dfef234bd507c5316b71aabb18306e6512b
        (base) jini1114@user1:~/git$ docker images
        REPOSITORY                       TAG       IMAGE ID       CREATED         SIZE
        <none>                           <none>    0e5831183a49   4 seconds ago   8.74GB
        ```
        
- 위험기상 예측모델 docker run
    - docker run -it --gpus all --ipc=host -e NVIDIA_VISIBLE_DEVICES=all -v /path/weather_prediction:/testdata/ --name final_test IMAGE_ID  /bin/bash
    
    ```python
    (base) jini1114@user1:~/git$ docker run -it --gpus all --ipc=host -e NVIDIA_VISIBLE_DEVICES=all -v /mnt/ai-nas02/WORK/jini1114/proof_of_validity/weather_prediction:/testdata/ --name final_test 0e5831183a49  /bin/bash
    ```
- 어노테이션 탐지모델 docker run
    - docker run -it --gpus all --ipc=host -e NVIDIA_VISIBLE_DEVICES=all -v /path/annotation:/testdata/ --name final_test IMAGE_ID  /bin/bash

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

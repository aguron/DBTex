docker build -t tomis:tomo3  \
  -f Dockerfile3 \
  --build-arg USER_NAME=$(whoami) \
  --build-arg USER_ID=$(id -u) \
  --build-arg GROUP_ID=$(id -g) ./

docker  run -it --ipc=host -p 8890:8888 -v /home/cmarasinou/:/workspace/ --mount type=bind,source=/mnt/titanxp/datasets/SPIE-Challenge/,target=/workspace/Data/ --mount type=bind,source=/data/breast/SPIEchallengeoutput/,target=/workspace/Output/   --gpus '"device=3"' tomis:tomo1


docker  run -it --ipc=host -p 8895:6007 -v /home/cmarasinou/:/workspace/ --mount type=bind,source=/mnt/titanxp/datasets/SPIE-Challenge/,target=/workspace/Data/ --mount type=bind,source=/data/breast/SPIEchallengeoutput/,target=/workspace/Output/   --gpus '"device=3"' tomis:tomo1


docker  run -it --ipc=host -p 8097:8097 -v /home/cmarasinou/Projects/newFasterRCNN/simple-faster-rcnn-pytorch/:/workspace/ --mount type=bind,source=/home/cmarasinou/Data/,target=/workspace/Data/ --gpus '"device=3"' tomis:tomo1


docker  run -it --ipc=host -p 8890:8888 -v /home/cmarasinou/:/workspace/ --mount type=bind,source=/mnt/titanxp/datasets/SPIE-Challenge/,target=/workspace/Data/ --mount type=bind,source=/data/breast/SPIEchallengeoutput/,target=/workspace/Output/ --mount type=bind,source=/mnt/titanxp/datasets/breast/,target=/workspace/Mount/   --gpus '"device=3"' tomis:tomo2




docker  run -it -p 8895:6006 -v /home/cmarasinou/Projects/tomochallenge/Faster-RCNN/runs/:/logs/ activeeon/tensorboard:latest
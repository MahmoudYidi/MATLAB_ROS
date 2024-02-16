clear all;
camera = webcam(1);
  
while true

picture = camera.snapshot;
testCNN(picture)
image(picture)
drawnow;
end
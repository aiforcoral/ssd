    # Real Time Detection
     
    # Importing the libraries
    import torch
    from torch.autograd import Variable
    import cv2
    from data import BaseTransform, VOC_CLASSES as labelmap
    from ssd import build_ssd
    import imageio
    # Defining a function that will do the detections
    def detect(frame, net, transform):
        height, width = frame.shape[:2]
        frame_t = transform(frame)[0]
        x = torch.from_numpy(frame_t).permute(2, 0, 1)
        x = Variable(x.unsqueeze(0))
        y = net(x)
        detections = y.data
        scale = torch.Tensor([width, height, width, height])
        # detections = [batch, number of classes, number of occurence, (score, x0, Y0, x1, y1)]
        for i in range(detections.size(1)):
            j = 0
            while detections[0, i, j, 0] >= 0.6:
                pt = (detections[0, i, j, 1:] * scale).numpy()
                cv2.rectangle(frame, (int(pt[0]), int(pt[1])), (int(pt[2]), int(pt[3])), (255, 0, 0), 2)
                cv2.putText(frame, labelmap[i - 1], (int(pt[0]), int(pt[1])), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
                j += 1
        return frame
     
    # Creating the SSD neural network
    net = build_ssd('test')
    net.load_state_dict(torch.load('ssd300_mAP_77.43_v2.pth', map_location = lambda storage, loc: storage))
     
    # Creating the transformation
    transform = BaseTransform(net.size, (104/256.0, 117/256.0, 123/256.0))
     
    # Doing some real time detection (assuming an external camera is connected to your computer)
    video_capture = cv2.VideoCapture(0) #0 for webcam 1 from external canera
    while True:
        _, frame = video_capture.read()
        canvas = detect(frame, net, transform)
        cv2.imshow('Video', canvas)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video_capture.release()
    cv2.destroyAllWindows()
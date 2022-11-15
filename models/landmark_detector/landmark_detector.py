import torch.nn as nn


class LandmarkDetectorModel(nn.Module):
    def __init__(self, config):
        super(LandmarkDetectorModel, self).__init__()
        self.mbnet = MobileNetV2(points_num=config.LANDMARK_POINTS)
        self.name = 'landmark_detector'
        self.iteration = 0
        self.config = config

        self.landmark_weights_path = os.path.join(config.PATH, self.name + '.pth')

        if len(config.GPU) > 1:
            self.mbnet = nn.DataParallel(self.mbnet, config.GPU)

        self.optimizer = optim.Adam(
            params=self.mbnet.parameters(),
            lr=self.config.LR,
            weight_decay=0.000001
        )


    def save(self):
        print('\nsaving %s...\n' % self.name)
        torch.save({
            'iteration': self.iteration,
            'detector': self.mbnet.state_dict()
        }, self.landmark_weights_path)

    def load(self):
        if os.path.exists(self.landmark_weights_path):
            print('Loading landmark detector...')

            if torch.cuda.is_available():
                data = torch.load(self.landmark_weights_path)
            else:
                data = torch.load(self.landmark_weights_path, map_location=lambda storage, loc: storage)

            self.mbnet.load_state_dict(data['detector'])
            self.iteration = data['iteration']
            print('Loading landmark detector complete!')

    def forward(self, images, masks):
        images_masked = images* (1 - masks).float() + masks

        landmark_gen = self.mbnet(images_masked)
        landmark_gen *= self.config.INPUT_SIZE

        return landmark_gen

    def process(self, images, masks, landmark_gt):
        self.iteration += 1
        self.optimizer.zero_grad()

        images_masked = images*(1-masks)+masks
        landmark_gen = self(images_masked, masks)
        landmark_gen = landmark_gen.reshape((-1, self.config.LANDMARK_POINTS, 2))
        loss = loss_landmark(landmark_gt.float(),landmark_gen, points_num=self.config.LANDMARK_POINTS)

        logs = [("loss", loss.item())]
        return landmark_gen, loss, logs

    def process_aug(self, images, masks, landmark_gt):
        self.optimizer.zero_grad()
        images_masked = images*(1-masks)+masks
        landmark_gen = self(images_masked, masks)
        landmark_gen = landmark_gen.reshape(-1,self.config.LANDMARK_POINTS,2)
        loss = loss_landmark(landmark_gt.float(),landmark_gen, points_num=self.config.LANDMARK_POINTS)

        logs = [("loss_aug", loss.item())]

        return landmark_gen, loss, logs



    def backward(self, loss):
        loss.backward()
        self.optimizer.step()

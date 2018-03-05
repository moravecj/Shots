import cxflow as cx
import numpy as np
import cv2


class ShotsPredict(cx.AbstractHook):
    results = []
    allResults = []
    file = '25011'
    idx = 0

    def before_training(self):
        cap = cv2.VideoCapture('C:/CXFLOW/Dataset/video_rai/' + self.file + '.mp4')
        fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        self.results = np.zeros(fc)

    def after_batch(self, stream_name, batch_data):
        bd = batch_data['predictions'][:]
        self.allResults.append(bd)
        for j in range(bd.shape[0]):
            y = bd[j]
            for k in range(y.shape[0]):
                self.results[k+self.idx] += y[k]
            self.idx += 1

    def after_epoch(self, epoch_id, epoch_data):
        np.save('D:/RAIDataset/video_rai/' + self.file + '_results_all.npy', self.allResults)
        np.save('D:/RAIDataset/video_rai/' + self.file + '_results.npy', self.results)
        print(self.results.shape)
        for i in range(self.results.shape[0]):
            if self.results[i] >= 90:
                self.results[i] = 1
            else:
                self.results[i] = 0

        last = 0

        with open('D:/RAIDataset/video_rai/' + self.file + '_my_results.txt', 'w') as f:
            f.write('0 ')
            for i in range(self.results.shape[0]):
                if last != self.results[i] and last == 0:
                    f.write('{}\n'.format(i-1))
                    last = self.results[i]
                if last != self.results[i] and last == 1:
                    ok = False
                    for k in range(1, 5):
                        if self.results[i+k] == 1:
                            ok = True
                    if not(ok):
                        f.write('{} '.format(i-1))
                        last = self.results[i]

            f.write('{}'.format(self.results.shape[0]-1))
import cxflow as cx
import numpy as np
import copy
import cv2


class ShotsPredict(cx.AbstractHook):
    results = []
    results_raw = []
    allResults = []
    file = '23553'
    idx = 0
    idx2 = 0
    fc = 0

    def before_training(self):
        cap = cv2.VideoCapture('C:/CXFLOW/Dataset/video_rai/' + self.file + '.mp4')
        self.fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        self.results = np.zeros(self.fc)
        self.results_raw = np.zeros(self.fc)

    def after_batch(self, stream_name, batch_data):
        bd = batch_data['predictions'][:]

        imgs = batch_data['images']
        img = imgs[0, 0, :, :, :]
        img *= 255
        cv2.imwrite('D:/outPy/' + batch_data['id'] + 'b.bmp', img)

        self.allResults.append(bd)
        for j in range(bd.shape[0]):
            y = bd[j]
            for k in range(y.shape[0]):
                self.results[k+self.idx] += y[k]
                if self.results[k+self.idx] > 90:
                    print(k+self.idx)
            self.idx += 1

    def after_epoch(self, epoch_id, epoch_data):
        np.save('D:/RAIDataset/video_rai/' + self.file + '_results_all.npy', self.allResults)
        np.save('D:/RAIDataset/video_rai/' + self.file + '_results.npy', self.results)
        np.save('D:/RAIDataset/video_rai/' + self.file + '_results_raw.npy', self.results_raw)

        pom = copy.deepcopy(self.results)
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
                            self.results[i] = 1
                    if not(ok):
                        f.write('{} '.format(i-1))
                        last = self.results[i]

            f.write('{}'.format(self.results.shape[0]-1))

        resultsLabels = np.zeros(self.fc)
        lastEn = -1

        self.results[:(self.results.shape[0] - 1)] = self.results[1:]

        with open('D:/RAIDataset/video_rai/23553_gt.txt') as f:
            for line in f:
                st = int(line.split()[0])
                en = int(line.split()[1])
                if lastEn == -1:
                    lastEn = en
                else:
                    resultsLabels[lastEn:st] = 1
                    lastEn = en

        idx = 0
        save_id = 0
        size_with_border = 80
        size_of_border = 5

        window = np.zeros((size_with_border * 10, size_with_border * 11, 3), dtype=np.uint8)

        for i in range(resultsLabels.shape[0]):
            if self.results[i] != resultsLabels[i]:
                print('{}'.format(i))
                k = 0
                for j in range(max(0, i - 5), min(resultsLabels.shape[0], i + 6)):
                    buf = cv2.imread('D:/RAIDataset/video_rai/' + self.file + '/' + str(j) + '.bmp')
                    buf = cv2.resize(buf, (size_with_border - 2 * size_of_border, size_with_border - 2 * size_of_border))

                    row, col = buf.shape[:2]

                    if self.results[j] == 1:
                        valPred = [0, 255, 0]
                    else:
                        valPred = [0, 0, 255]

                    if resultsLabels[j] == 1:
                        valLab = [0, 255, 0]
                    else:
                        valLab = [0, 0, 255]

                    border = cv2.copyMakeBorder(buf, top=size_of_border, bottom=size_of_border, left=0,
                                                right=0, borderType=cv2.BORDER_CONSTANT,
                                                value=valPred)

                    border = cv2.copyMakeBorder(border, top=0, bottom=0, left=size_of_border,
                                                right=size_of_border, borderType=cv2.BORDER_CONSTANT,
                                                value=valLab)

                    window[(idx * size_with_border):(idx * size_with_border + size_with_border),
                        (k * size_with_border):(k * size_with_border + size_with_border), :] = border

                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(window, str(pom[j+1]), ((k * size_with_border + 10), (idx * size_with_border +
                                (size_with_border - 10))), font, 0.3, (255, 255, 255),
                                1, cv2.LINE_AA)

                    k += 1

                idx += 1
                if idx == 10:
                    cv2.imwrite('D:/outPy/predict/{}.jpg'.format(save_id), window)
                    window = np.zeros((size_with_border * 10, size_with_border * 11, 3), dtype=np.uint8)
                    save_id += 1
                    idx = 0
        if idx != 0:
            cv2.imwrite('D:/outPy/predict/{}.jpg'.format(save_id), window)
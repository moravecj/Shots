import os
import numpy as np
import cxflow as cx
import cv2
import random
import copy


class ShotsDataset(cx.BaseDataset):
    def __add_image_and_label_to_batch(self, img, lab) -> bool:
        img = np.array(img, dtype=np.float32)
        img /= 255
        if 0 <= self._frame < self._batch_size - 1:
            self._images[self._frame + 1, :, :, 0:(self._num_of_frames - 1), :] = \
                self._images[self._frame, :, :, 1:self._num_of_frames, :]

            self._images[self._frame + 1, :, :, self._num_of_frames - 1, :] = img

            self._labels[self._frame + 1, 0:(self._num_of_frames - 1)] = \
                self._labels[self._frame, 1:self._num_of_frames]

            self._labels[self._frame + 1, self._num_of_frames - 1] = lab

            if self._frame == self._batch_size - 2:
                self._frame = self._frame + 1
                """
                for i in range(0, self._batch_size):
                    for j in range(0, self._num_of_frames):
                        im = self._images[i, :, :, j, :]
                        im2 = cv2.resize(im, (320, 320))
                        cv2.imshow('fig1',  im2)
                        print(self._labels[i, j])
                        cv2.waitKey(0)
                """
                return True
            self._frame = self._frame + 1

        elif self._frame == self._batch_size - 1:
            self._images[0, :, :, 0:(self._num_of_frames - 1), :] = \
                self._images[self._batch_size - 1, :, :, 1:self._num_of_frames, :]

            self._images[0, :, :, self._num_of_frames - 1, :] = img

            self._labels[0, 0:(self._num_of_frames - 1)] = \
                self._labels[self._batch_size - 1, 1:self._num_of_frames]

            self._labels[0, self._num_of_frames - 1] = lab

            self._frame = 0
        else:
            self._images[0, :, :, self._frame + self._num_of_frames , :] = img
            self._labels[0, self._frame + self._num_of_frames ] = lab
            self._frame = self._frame + 1
        return False

    def __read_labels(self) -> None:
        idx = 0
        for fn in os.listdir(os.path.join(self._data_root, 'labels')):
            x = []
            start = 1
            with open(os.path.join(self._data_root, 'labels', fn)) as f:
                for line in f:
                    curr = int(line) + 1
                    if curr != start:
                        x.append([start - 1, curr - 2])
                    start = curr + 1

            cap = cv2.VideoCapture(self._videos_dir[idx])
            fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self._count = self._count + fc - 1
            if len(x) > 0:
                x.append([start - 2, fc - 1])
            else:
                x.append([start - 1, fc - 1])
            self._labels_dir.append(x)
            idx = idx + 1

    def _configure_dataset(self, data_root='Dataset', batch_size: int=50, num_of_frames: int=32,
                           length_of_fadein: int = 10, **kwargs) -> None:
        self._batch_size = batch_size
        self._data_root = data_root
        self._data = {}
        self._videos_dir = [f.path for f in os.scandir(os.path.join(data_root, 'TRECVidSubset100')) if f.is_file()]
        self._labels_dir = []
        self._count = 0
        self.__read_labels()
        self._num_of_frames = num_of_frames
        self._frame = -self._num_of_frames
        self._images = np.zeros((self._batch_size, 32, 32, self._num_of_frames, 3), dtype=np.float32)
        self._labels = np.zeros((self._batch_size, self._num_of_frames), dtype=np.float32)
        self._count_in_batch = self._batch_size * self._num_of_frames
        self._length_of_fadein = length_of_fadein
        self._perm = np.random.permutation(len(self._labels_dir))
        self._max_shot_length = 100

    def train_stream(self) -> cx.Stream:
        self._frame = -self._num_of_frames
        self._train = self._perm[:80]

        videos_done = 0

        pom_labels = copy.deepcopy(self._labels_dir)

        index = random.randint(0, len(self._train) - 1)
        x = self._train[index]
        s = random.randint(0, len(pom_labels[x]) - 1)

        shot = pom_labels[x][s]
        shot[1] = min(shot[1], self._max_shot_length)
        pom_labels[x].pop(s)
        if len(pom_labels[x]) == 0:
            self._train = np.delete(self._train, index)
            videos_done = videos_done + 1
            print('Done video: ', videos_done)

        cap = cv2.VideoCapture(self._videos_dir[x])

        idx = 0
        i = 0
        while True:
            if i % 10000 == 0:
                print(i)
            i = i + 1
            if shot[0] + idx < shot[1]:
                cap.set(1, shot[0] + idx)
                ret, buf = cap.read()
                buf = cv2.resize(buf, (32, 32))
                #self.__add_image_and_label(buf, 0)

                if self.__add_image_and_label_to_batch(buf, 0):
                    yield {'images': self._images, 'labels': self._labels}

                idx = idx + 1
            elif shot[0] + idx == shot[1]:
                cap.set(1, shot[0] + idx)
                ret, fr1 = cap.read()
                fr1 = cv2.resize(fr1, (32, 32))
                if random.random() > 0.1:
                    #self.__add_image_and_label(fr1, 1)

                    if self.__add_image_and_label_to_batch(fr1, 1):
                        yield {'images': self._images, 'labels': self._labels}

                    idx = idx + 1
                else:
                    """
                    x = random.choice([j for j in self._train if j not in [x]])
                    x = random.choice(self._train)
                    s = random.randint(0, len(self._labels_dir[x]) - 1)
                    """
                    if len(self._train) == 0:
                        break

                    index = random.randint(0, len(self._train) - 1)
                    x = self._train[index]
                    s = random.randint(0, len(pom_labels[x]) - 1)
                    shot[1] = min(shot[1], self._max_shot_length)
                    shot = pom_labels[x][s]

                    pom_labels[x].pop(s)

                    if len(pom_labels[x]) == 0:
                        self._train = np.delete(self._train, index)
                        videos_done = videos_done + 1
                        print('Done video: ', videos_done)

                    idx = 0

                    cap = cv2.VideoCapture(self._videos_dir[x])

                    cap.set(1, shot[0] + idx)
                    ret, fr2 = cap.read()
                    fr2 = cv2.resize(fr2, (32, 32))
                    for IN in range(0, self._length_of_fadein):
                        fadein = IN / float(self._length_of_fadein)
                        dst = cv2.addWeighted(fr1, 1 - fadein, fr2, fadein, 0)
                        dst = cv2.resize(dst, (32, 32))
                        if IN < 3 or IN >= self._length_of_fadein - 2:
                            #self.__add_image_and_label(dst, 0)
                            if self.__add_image_and_label_to_batch(dst, 0):
                                yield {'images': self._images, 'labels': self._labels}
                        else:
                            #self.__add_image_and_label(dst, 1)
                            if self.__add_image_and_label_to_batch(dst, 1):
                                yield {'images': self._images, 'labels': self._labels}
            else:
                """
                x = random.choice([j for j in self._train if j not in [x]])
                s = random.randint(0, len(self._labels_dir[x]) - 1)
                """
                if len(self._train) == 0:
                    break

                index = random.randint(0, len(self._train) - 1)
                x = self._train[index]
                s = random.randint(0, len(pom_labels[x]) - 1)

                shot = pom_labels[x][s]
                shot[1] = min(shot[1], self._max_shot_length)

                pom_labels[x].pop(s)

                if len(pom_labels[x]) == 0:
                    self._train = np.delete(self._train, index)
                    videos_done = videos_done + 1
                    print('Done video: ', videos_done)


                idx = 0

                cap = cv2.VideoCapture(self._videos_dir[x])

    def test_stream(self) -> cx.Stream:
        self._frame = -self._num_of_frames

        self._test = self._perm[80:]

        pom_labels = copy.deepcopy(self._labels_dir)

        index = random.randint(0, len(self._test) - 1)
        x = self._test[index]
        s = random.randint(0, len(pom_labels[x]) - 1)

        shot = pom_labels[x][s]
        shot[1] = min(shot[1], self._max_shot_length)

        pom_labels[x].pop(s)
        if len(pom_labels[x]) == 0:
            self._test = np.delete(self._test, index)

        cap = cv2.VideoCapture(self._videos_dir[x])

        idx = 0
        for i in range(0, self._count - self._num_of_frames):
            if shot[0] + idx < shot[1]:
                cap.set(1, shot[0] + idx)
                ret, buf = cap.read()
                buf = cv2.resize(buf, (32, 32))
                if self.__add_image_and_label_to_batch(buf, 0):
                    yield {'images': self._images, 'labels': self._labels}
                idx = idx + 1
            elif shot[0] + idx == shot[1]:
                cap.set(1, shot[0] + idx)
                ret, fr1 = cap.read()
                fr1 = cv2.resize(fr1, (32, 32))
                if random.random() > 0.1:
                    if self.__add_image_and_label_to_batch(fr1, 1):
                        yield {'images': self._images, 'labels': self._labels}
                    idx = idx + 1
                else:
                    if len(self._test) == 0:
                        break

                    index = random.randint(0, len(self._test) - 1)
                    x = self._test[index]
                    s = random.randint(0, len(pom_labels[x]) - 1)

                    shot = pom_labels[x][s]
                    shot[1] = min(shot[1], self._max_shot_length)

                    pom_labels[x].pop(s)
                    if len(pom_labels[x]) == 0:
                        self._test = np.delete(self._test, index)

                    idx = 0

                    cap = cv2.VideoCapture(self._videos_dir[x])

                    cap.set(1, shot[0] + idx)
                    ret, fr2 = cap.read()
                    fr2 = cv2.resize(fr2, (32, 32))
                    for IN in range(0, self._length_of_fadein):
                        fadein = IN / float(self._length_of_fadein)
                        dst = cv2.addWeighted(fr1, 1 - fadein, fr2, fadein, 0)
                        if IN < 3 or IN >= self._length_of_fadein - 2:
                            if self.__add_image_and_label_to_batch(dst, 0):
                                yield {'images': self._images, 'labels': self._labels}
                        else:
                            if self.__add_image_and_label_to_batch(dst, 1):
                                yield {'images': self._images, 'labels': self._labels}
            else:
                if len(self._test) == 0:
                    break

                index = random.randint(0, len(self._test) - 1)
                x = self._test[index]
                s = random.randint(0, len(pom_labels[x]) - 1)

                shot = pom_labels[x][s]
                shot[1] = min(shot[1], self._max_shot_length)

                pom_labels[x].pop(s)
                if len(pom_labels[x]) == 0:
                    self._test = np.delete(self._test, index)

                idx = 0

                cap = cv2.VideoCapture(self._videos_dir[x])
    #def test_stream(self) -> cx.Stream:
    #def download(self) -> None:
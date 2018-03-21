import logging

import os
import numpy as np
import cxflow as cx
import cv2
import random
import copy
import time


class ShotsDataset(cx.BaseDataset):
    def __new_shot(self, pom_labels, train):
        if train:
            index = random.randint(0, len(self._train) - 1)
            x = self._train[index]
        else:
            index = random.randint(0, len(self._test) - 1)
            x = self._test[index]
        s = random.randint(0, len(pom_labels[x]) - 1)

        shot = pom_labels[x][s]
        leng = random.randint(10, self._max_shot_length)
        if shot[1] - shot[0] > leng:
            shot[0] = random.randint(shot[0], shot[1] - leng)
        shot[1] = shot[0] + min(shot[1] - shot[0], leng)

        pom_labels[x].pop(s)

        if len(pom_labels[x]) == 0:
            if train:
                self._train = np.delete(self._train, index)
            else:
                self._test = np.delete(self._test, index)
        return shot, x, pom_labels

    def __number_of_shots_left(self, lbls) -> int:
        count = 0
        for i in self._train:
            count += len(lbls[i])
        return count
    def __fill_vectors_from_frame(self, beg):
        ind = self._frames_remember

        self._images[beg, 0:ind, :, :, :] = \
            self._images[self._frame - 1, (self._num_of_frames - ind):, :, :, :]
        self._labels[beg, 0:ind] = \
            self._labels[self._frame - 1, (self._num_of_frames - ind):]

        self._frames_needed = self._num_of_frames - ind

    def __fill_vectors_from_frame_without_labels(self, beg):
        ind = self._frames_remember

        self._images[beg, 0:ind, :, :, :] = \
            self._images[self._frame - 1, (self._num_of_frames - ind):, :, :, :]

        self._frames_needed = self._num_of_frames - ind

    def __add_image_and_label_to_batch(self, img, lab) -> bool:
        img = np.array(img, dtype=np.float32)
        img /= 255

        #if lab == 0 and random.random() < 0.1:
        #    img = img + random.uniform(0, 0.3)

        if self._frame == self._batch_size:
            self.__fill_vectors_from_frame(0)
            self._frame = 0

        if self._frames_needed > 1:
            self._images[self._frame, self._num_of_frames - self._frames_needed, :, :, :] = img
            self._labels[self._frame, self._num_of_frames - self._frames_needed] = lab
            self._frames_needed -= 1
        if self._frames_needed == 1:
            self._images[self._frame, self._num_of_frames - self._frames_needed, :, :, :] = img
            self._labels[self._frame, self._num_of_frames - self._frames_needed] = lab
            self._frames_needed -= 1

            self._frame += 1

            if self._frame == self._batch_size:
                return True
            else:
                self.__fill_vectors_from_frame(self._frame)
        return False

    def __add_image_only(self, img) -> bool:
        img = np.array(img, dtype=np.float32)
        img /= 255

        if self._frame == self._batch_size:
            self.__fill_vectors_from_frame_without_labels(0)
            self._frame = 0

        if self._frames_needed > 1:
            self._images[self._frame, self._num_of_frames - self._frames_needed, :, :, :] = img
            self._frames_needed -= 1
        if self._frames_needed == 1:
            self._images[self._frame, self._num_of_frames - self._frames_needed, :, :, :] = img
            self._frames_needed -= 1
            self._frame += 1

            if self._frame == self._batch_size:
                return True
            else:
                self.__fill_vectors_from_frame_without_labels(self._frame)
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
            cap.release()
            self._count = self._count + fc - 1
            if len(x) > 0:
                x.append([start - 2, fc - 1])
            else:
                x.append([start - 1, fc - 1])
            self._labels_dir.append(x)
            idx = idx + 1

    def _configure_dataset(self, data_root='Dataset', batch_size: int=50, num_of_frames: int=100,
                           length_of_fadein: int = 10, size_of_pictures: int = 32, **kwargs) -> None:
        self._batch_size = batch_size
        self._data_root = data_root
        self._videos_dir = [f.path for f in os.scandir(os.path.join(data_root, 'TRECVidSubset100')) if f.is_file()]
        self._labels_dir = []
        self._count = 0
        self.__read_labels()
        self._num_of_frames = num_of_frames
        self._frame = 0
        self._size_of_pictures = size_of_pictures
        self._images = np.zeros((self._batch_size, self._num_of_frames, self._size_of_pictures, self._size_of_pictures, 3), dtype=np.float32)
        self._labels = np.zeros((self._batch_size, self._num_of_frames), dtype=np.int64)
        self._count_in_batch = self._batch_size * self._num_of_frames
        self._length_of_fadein = 50
        self._perm = np.random.permutation(len(self._labels_dir))
        self._max_shot_length = 30
        self._black_frame = np.zeros((self._size_of_pictures, self._size_of_pictures, 3), dtype=np.uint8)
        self._frames_needed = self._num_of_frames
        self._frames_remember = self._num_of_frames - 1
        self._dat_index = 0

    def train_stream(self) -> cx.Stream:
        #self._frame = -self._num_of_frames
        self._frame = 0
        self._frames_needed = self._num_of_frames
        self._train = self._perm[:85]

        pom_labels = copy.deepcopy(self._labels_dir)

        index = random.randint(0, len(self._train) - 1)
        x = self._train[index]
        s = random.randint(0, len(pom_labels[x]) - 1)

        shot = pom_labels[x][s]
        leng = random.randint(10, self._max_shot_length)
        if shot[1] - shot[0] > leng:
            shot[0] = random.randint(shot[0], shot[1] - leng)
        shot[1] = shot[0] + min(shot[1] - shot[0], leng)

        pom_labels[x].pop(s)
        if len(pom_labels[x]) == 0:
            self._train = np.delete(self._train, index)

        cap = cv2.VideoCapture(self._videos_dir[x])

        idx = 0
        i = 0
        while True:
            i = i + 1
            if shot[0] + idx < shot[1]:
                cap.set(1, shot[0] + idx)
                ret, buf = cap.read()
                buf = cv2.resize(buf, (self._size_of_pictures, self._size_of_pictures))

                if self.__add_image_and_label_to_batch(buf, 0):
                    yield {'images': self._images, 'labels': self._labels}

                idx = idx + 1
            elif shot[0] + idx == shot[1]:
                cap.set(1, shot[0] + idx)
                ret, fr1 = cap.read()
                fr1 = cv2.resize(fr1, (self._size_of_pictures, self._size_of_pictures))
                choice = random.random()
                if choice >= 0.5:
                    if self.__add_image_and_label_to_batch(fr1, 1):
                        yield {'images': self._images, 'labels': self._labels}
                    idx = idx + 1

                    if choice < 0.75:
                        if len(self._train) == 0:
                            break

                        shot, x, pom_labels = self.__new_shot(pom_labels, True)

                        idx = 0

                        cap.release()
                        cap = cv2.VideoCapture(self._videos_dir[x])

                        cap.set(1, shot[0] + idx)
                        ret, fr2 = cap.read()
                        fr2 = cv2.resize(fr2, (self._size_of_pictures, self._size_of_pictures))
                        length_of_fadein = random.randint(10, self._length_of_fadein)
                        for IN in range(0, length_of_fadein):
                            fadein = IN / float(length_of_fadein)
                            dst = cv2.addWeighted(self._black_frame, 1 - fadein, fr2, fadein, 0)
                            dst = cv2.resize(dst, (self._size_of_pictures, self._size_of_pictures))
                            if self.__add_image_and_label_to_batch(dst, 1):
                                yield {'images': self._images, 'labels': self._labels}

                elif choice >= 0.25:
                    if self.__add_image_and_label_to_batch(fr1, 1):
                        yield {'images': self._images, 'labels': self._labels}

                    idx = idx + 1
                    length_of_fadein = random.randint(10, self._length_of_fadein)
                    for IN in range(1, length_of_fadein + 1):
                        fadein = IN / float(length_of_fadein)
                        dst = cv2.addWeighted(fr1, 1 - fadein, self._black_frame, fadein, 0)
                        dst = cv2.resize(dst, (self._size_of_pictures, self._size_of_pictures))

                        if self.__add_image_and_label_to_batch(dst, 1):
                            yield {'images': self._images, 'labels': self._labels}
                else:
                    if len(self._train) == 0:
                        break

                    shot, x, pom_labels = self.__new_shot(pom_labels, True)
                    idx = 0

                    cap.release()
                    cap = cv2.VideoCapture(self._videos_dir[x])

                    cap.set(1, shot[0] + idx)
                    ret, fr2 = cap.read()
                    fr2 = cv2.resize(fr2, (self._size_of_pictures, self._size_of_pictures))
                    length_of_fadein = random.randint(10, self._length_of_fadein)
                    for IN in range(0, length_of_fadein + 1):
                        fadein = IN / float(length_of_fadein)
                        dst = cv2.addWeighted(fr1, 1 - fadein, fr2, fadein, 0)
                        dst = cv2.resize(dst, (self._size_of_pictures, self._size_of_pictures))
                        if self.__add_image_and_label_to_batch(dst, 1):
                            yield {'images': self._images, 'labels': self._labels}

            else:
                if len(self._train) == 0:
                    break

                shot, x, pom_labels = self.__new_shot(pom_labels, True)

                idx = 0

                cap.release()
                cap = cv2.VideoCapture(self._videos_dir[x])

    def test_stream(self) -> cx.Stream:
        self._frame = 0
        self._frames_needed = self._num_of_frames
        self._test = self._perm[85:]
        
        pom_labels = copy.deepcopy(self._labels_dir)

        index = random.randint(0, len(self._test) - 1)
        x = self._test[index]
        s = random.randint(0, len(pom_labels[x]) - 1)

        shot = pom_labels[x][s]
        leng = random.randint(10, self._max_shot_length)
        if shot[1] - shot[0] > leng:
            shot[0] = random.randint(shot[0], shot[1] - leng)
        shot[1] = shot[0] + min(shot[1] - shot[0], leng)

        pom_labels[x].pop(s)
        if len(pom_labels[x]) == 0:
            self._test = np.delete(self._test, index)

        cap = cv2.VideoCapture(self._videos_dir[x])

        idx = 0
        i = 0
        while True:
            i = i + 1
            if shot[0] + idx < shot[1]:
                cap.set(1, shot[0] + idx)
                ret, buf = cap.read()
                buf = cv2.resize(buf, (self._size_of_pictures, self._size_of_pictures))
                if self.__add_image_and_label_to_batch(buf, 0):
                    yield {'images': self._images, 'labels': self._labels}
                idx = idx + 1

            elif shot[0] + idx == shot[1]:
                cap.set(1, shot[0] + idx)
                ret, fr1 = cap.read()
                fr1 = cv2.resize(fr1, (self._size_of_pictures, self._size_of_pictures))
                choice = random.random()
                if choice >= 0.5:
                    if self.__add_image_and_label_to_batch(fr1, 1):
                        yield {'images': self._images, 'labels': self._labels}
                    idx = idx + 1

                    if choice < 0.75:
                        if len(self._test) == 0:
                            break

                        shot, x, pom_labels = self.__new_shot(pom_labels, False)

                        idx = 0

                        cap.release()
                        cap = cv2.VideoCapture(self._videos_dir[x])

                        cap.set(1, shot[0] + idx)
                        ret, fr2 = cap.read()
                        fr2 = cv2.resize(fr2, (self._size_of_pictures, self._size_of_pictures))
                        length_of_fadein = random.randint(10, self._length_of_fadein)
                        for IN in range(0, length_of_fadein):
                            fadein = IN / float(length_of_fadein)
                            dst = cv2.addWeighted(self._black_frame, 1 - fadein, fr2, fadein, 0)
                            dst = cv2.resize(dst, (self._size_of_pictures, self._size_of_pictures))
                            if self.__add_image_and_label_to_batch(dst, 1):
                                yield {'images': self._images, 'labels': self._labels}

                elif choice >= 0.25:
                    if self.__add_image_and_label_to_batch(fr1, 1):
                        # print(self._images.shape, ' ', self._labels.shape)
                        yield {'images': self._images, 'labels': self._labels}

                    idx = idx + 1
                    length_of_fadein = random.randint(10, self._length_of_fadein)
                    for IN in range(1, length_of_fadein + 1):
                        fadein = IN / float(length_of_fadein)
                        dst = cv2.addWeighted(fr1, 1 - fadein, self._black_frame, fadein, 0)
                        dst = cv2.resize(dst, (self._size_of_pictures, self._size_of_pictures))

                        if self.__add_image_and_label_to_batch(dst, 1):
                            yield {'images': self._images, 'labels': self._labels}

                else:
                    if len(self._test) == 0:
                        break

                    shot, x, pom_labels = self.__new_shot(pom_labels, False)

                    idx = 0

                    cap.release()
                    cap = cv2.VideoCapture(self._videos_dir[x])

                    cap.set(1, shot[0] + idx)
                    ret, fr2 = cap.read()
                    fr2 = cv2.resize(fr2, (self._size_of_pictures, self._size_of_pictures))
                    length_of_fadein = random.randint(10, self._length_of_fadein)
                    for IN in range(0, length_of_fadein + 1):
                        fadein = IN / float(length_of_fadein)
                        dst = cv2.addWeighted(fr1, 1 - fadein, fr2, fadein, 0)
                        dst = cv2.resize(dst, (self._size_of_pictures, self._size_of_pictures))
                        if self.__add_image_and_label_to_batch(dst, 1):
                            yield {'images': self._images, 'labels': self._labels}
            else:
                if len(self._test) == 0:
                    break

                shot, x, pom_labels = self.__new_shot(pom_labels, False)

                idx = 0

                cap.release()
                cap = cv2.VideoCapture(self._videos_dir[x])


    def predict_stream(self) -> cx.Stream:
        file = '23553'

        cap = cv2.VideoCapture('D:/RAIDataset/video_rai/' + file + '.mp4')
        fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        idx = 0
        start = time.time()
        bid = 0
        while idx < fc:
            buf = cv2.imread('D:/RAIDataset/video_rai/' + file + '/' + str(idx) + '.bmp')
            #cap.set(1, idx)
            #ret, buf = cap.read()
            #buf = cv2.resize(buf,(32,32))

            if self.__add_image_only(buf):
                img = copy.deepcopy(self._images[0, 0, :, :, :])
                img *= 255
                cv2.imwrite('D:/outPy/' + str(bid) + 'a.bmp', img)

                yield {'images': self._images, 'id': str(bid)}
                bid += 1

            idx += 1

        end = time.time()
        logging.info(end - start)

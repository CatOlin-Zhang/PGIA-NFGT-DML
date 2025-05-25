import numpy as np

class MNISTImageReader:
    _expected_magic = 2051
    _current_index = 0

    def __init__(self, path):
        if not path.endswith('.idx3-ubyte'):
            raise NameError("File must be a '.idx3-ubyte' extension")
        self.__path = path
        self.__file_object = None
        self._num_of_images = 0
        self._num_of_rows = 0
        self._num_of_cols = 0

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def open(self):
        try:
            self.__file_object = open(self.__path, 'rb')
        except IOError as e:
            print(f"Failed to open file {self.__path}: {e}")
            raise

        magic_number = int.from_bytes(self.__file_object.read(4), byteorder='big')
        if magic_number != self._expected_magic:
            raise TypeError("The File is not a properly formatted .idx3-ubyte file!")

        self._num_of_images = int.from_bytes(self.__file_object.read(4), byteorder='big')
        print('Total {} images ...'.format(self._num_of_images))
        self._num_of_rows = int.from_bytes(self.__file_object.read(4), byteorder='big')
        self._num_of_cols = int.from_bytes(self.__file_object.read(4), byteorder='big')

    def close(self):
        if self.__file_object:
            self.__file_object.close()

    def read(self, num=None):
        if num is None:
            num = self._num_of_images - self._current_index

        feasible_num = min(num, self._num_of_images - self._current_index)
        raw_image_data = self.__file_object.read(self._num_of_rows * self._num_of_cols * feasible_num)
        index = range(self._current_index + 1, self._current_index + feasible_num + 1)
        images = np.frombuffer(raw_image_data, dtype=np.uint8).reshape(
            (feasible_num, self._num_of_rows, self._num_of_cols))
        # Normalize and add channel dimension
        images = images.astype(np.float32) / 255.0
        images = np.expand_dims(images, axis=1)  # Shape: (num_samples, 1, 28, 28)
        self._current_index += feasible_num
        return index, images

    @property
    def num_of_images(self):
        return self._num_of_images


class MNISTLabelReader:
    _expected_magic = 2049
    _current_index = 0

    def __init__(self, path):
        if not path.endswith('.idx1-ubyte'):
            raise NameError("File must be a '.idx1-ubyte' extension")
        self.__file_path = path
        self.__file_object = None
        self._num_of_labels = 0

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def open(self):
        try:
            self.__file_object = open(self.__file_path, 'rb')
        except IOError as e:
            print(f"Failed to open file {self.__file_path}: {e}")
            raise

        magic_number = int.from_bytes(self.__file_object.read(4), byteorder='big')
        if magic_number != self._expected_magic:
            raise TypeError("The File is not a properly formatted .idx1-ubyte file!")
        self._num_of_labels = int.from_bytes(self.__file_object.read(4), byteorder='big')
        print('Total {} labels ...'.format(self._num_of_labels))

    def close(self):
        if self.__file_object:
            self.__file_object.close()

    def read(self, num=None):
        if num is None:
            num = self._num_of_labels - self._current_index

        feasible_num = min(num, self._num_of_labels - self._current_index)
        raw_label_data = self.__file_object.read(feasible_num)
        index = range(self._current_index + 1, self._current_index + feasible_num + 1)
        labels = np.frombuffer(raw_label_data, dtype=np.uint8).reshape((feasible_num,))
        self._current_index += feasible_num
        return index, labels

    @property
    def num_of_labels(self):
        return self._num_of_labels




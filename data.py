# author: enijkamp@ucla.edu

import os
import pickle
import numpy as np
import torch
import torch.utils.data
import PIL

class IgnoreLabelDataset(torch.utils.data.Dataset):
    def __init__(self, orig):
        self.orig = orig

    def __getitem__(self, index):
        return self.orig[index][0]

    def __len__(self):
        return len(self.orig)

# author: enijkamp@ucla.edu

import os
import pickle
import numpy as np
import torch
import torch.utils.data
import PIL

class IgnoreLabelDataset(torch.utils.data.Dataset):
    def __init__(self, orig):
        self.orig = orig

    def __getitem__(self, index):
        return self.orig[index][0]

    def __len__(self):
        return len(self.orig)


class SingleImagesFolderMTDataset(torch.utils.data.Dataset):
    def __init__(self, root, cache, transform=None, workers=8, split_size=200, protocol=None, num_images=None):
        if cache is not None and os.path.exists(cache):
            with open(cache, 'rb') as f:
                self.images = pickle.load(f)
        else:
            self.transform = transform if not transform is None else lambda x: x
            self.images = []

            def split_seq(seq, size):
                newseq = []
                splitsize = 1.0 / size * len(seq)
                for i in range(size):
                    newseq.append(seq[int(round(i * splitsize)):int(round((i + 1) * splitsize))])
                return newseq

            def map(path_imgs):
                imgs_0 = [self.transform(np.array(PIL.Image.open(os.path.join(root, p_i)))) for p_i in path_imgs]
                imgs_1 = [self.compress(img) for img in imgs_0]

                print('.')
                return imgs_1

            path_imgs = os.listdir(root)
            if num_images:
                path_imgs = path_imgs[:num_images]

            n_splits = len(path_imgs) // split_size
            path_imgs_splits = split_seq(path_imgs, n_splits)

            from multiprocessing.dummy import Pool as ThreadPool
            pool = ThreadPool(workers)
            results = pool.map(map, path_imgs_splits)
            pool.close()
            pool.join()

            for r in results:
                self.images.extend(r)

            if cache is not None:
                with open(cache, 'wb') as f:
                    pickle.dump(self.images, f, protocol=protocol)

        print('Total number of images {}'.format(len(self.images)))

    def __getitem__(self, item):
        return self.decompress(self.images[item])

    def __len__(self):
        return len(self.images)

    @staticmethod
    def compress(img):
        return img

    @staticmethod
    def decompress(output):
        return output


class SingleImagesFolderCompressedMTDataset(torch.utils.data.Dataset):
    def __init__(self, root, cache, transform=None, transform2=None, workers=8, split_size=10, num_images=None):
        self.transform2 = transform2
        if cache is not None and os.path.exists(cache):
            with open(cache, 'rb') as f:
                self.images = pickle.load(f)
        else:
            self.transform = transform if not transform is None else lambda x: x
            self.images = []

            def split_seq(seq, size):
                newseq = []
                splitsize = 1.0 / size * len(seq)
                for i in range(size):
                    newseq.append(seq[int(round(i * splitsize)):int(round((i + 1) * splitsize))])
                return newseq

            def map(path_imgs):
                imgs_0 = [self.transform(np.array(PIL.Image.open(os.path.join(root, p_i)))) for p_i in path_imgs]
                imgs_1 = [self.compress(img) for img in imgs_0]

                print('.')
                return imgs_1

            path_imgs = [f for f in os.listdir(root) if any(f.endswith(e) for e in ['png', 'jpg', 'bmp'])]
            if num_images:
                path_imgs = path_imgs[:num_images]

            n_splits = len(path_imgs) // split_size
            path_imgs_splits = split_seq(path_imgs, n_splits)

            from multiprocessing.dummy import Pool as ThreadPool
            pool = ThreadPool(workers)
            results = pool.map(map, path_imgs_splits)
            pool.close()
            pool.join()

            for r in results:
                self.images.extend(r)

            if cache is not None:
                with open(cache, 'wb') as f:
                    pickle.dump(self.images, f)

        print('Total number of images {}'.format(len(self.images)))

    def __getitem__(self, item):
        return self.transform2(self.decompress(self.images[item]))

    def __len__(self):
        return len(self.images)

    @staticmethod
    def compress(img):
        import torchvision.transforms.functional as F
        img_p = F.to_pil_image(img)
        import io
        output = io.BytesIO()
        img_p.save(output, 'JPEG')
        return output

    @staticmethod
    def decompress(output):
        output.seek(0)
        return PIL.Image.open(output)



class CelebDataset(torch.utils.data.Dataset):
    def __init__(self, root, cache, transform=None, workers=32, protocol=None):
        if cache is not None and os.path.exists(cache):
            with open(cache, 'rb') as f:
                images = pickle.load(f)
                if isinstance(images, list):
                    images = torch.stack(images, dim=0)
                self.images = ((images / 255.) * 2.) - 1.
        else:
            self.transform = transform if not transform is None else lambda x: x
            self.images = []

            def split_seq(seq, size):
                newseq = []
                splitsize = 1.0 / size * len(seq)
                for i in range(size):
                    newseq.append(seq[int(round(i * splitsize)):int(round((i + 1) * splitsize))])
                return newseq

            def map(path_imgs):
                imgs_0 = [self.transform(np.array(PIL.Image.open(os.path.join(root, p_i)))) for p_i in path_imgs]
                imgs_1 = [self.compress(img) for img in imgs_0]

                print('.')
                return imgs_1

            path_imgs = os.listdir(root)
            n_splits = len(path_imgs) // 1000
            path_imgs_splits = split_seq(path_imgs, n_splits)

            from multiprocessing.dummy import Pool as ThreadPool
            pool = ThreadPool(workers)
            results = pool.map(map, path_imgs_splits)
            pool.close()
            pool.join()

            for r in results:
                self.images.extend(r)

            if cache is not None:
                with open(cache, 'wb') as f:
                    pickle.dump(self.images, f, protocol=protocol)

        print('Total number of images {}'.format(len(self.images)))

    def __getitem__(self, item):
        return self.decompress(self.images[item])

    def __len__(self):
        return len(self.images)

    @staticmethod
    def compress(img):
        return img

    @staticmethod
    def decompress(output):
        return output


class SingleImagesCompressedMTDatasetWrap(torch.utils.data.Dataset):
    def __init__(self, ds, cache, transform=None, transform2=None, workers=32):
        self.transform2 = transform2
        if cache is not None and os.path.exists(cache):
            with open(cache, 'rb') as f:
                self.images = pickle.load(f)
        else:
            self.transform = transform if not transform is None else lambda x: x
            self.images = []

            def split_seq(seq, size):
                newseq = []
                splitsize = 1.0 / size * len(seq)
                for i in range(size):
                    newseq.append(seq[int(round(i * splitsize)):int(round((i + 1) * splitsize))])
                return newseq

            def map(index_imgs):
                imgs_0 = [self.transform(ds[p_i]) for p_i in index_imgs]
                imgs_1 = [self.compress(img) for img in imgs_0]

                print('.')
                return imgs_1

            index_imgs = [i for i in range(len(ds))]
            n_splits = len(index_imgs) // 1000
            index_imgs_splits = split_seq(index_imgs, n_splits)

            from multiprocessing.dummy import Pool as ThreadPool
            pool = ThreadPool(workers)
            results = pool.map(map, index_imgs_splits)
            pool.close()
            pool.join()

            for r in results:
                self.images.extend(r)

            if cache is not None:
                with open(cache, 'wb') as f:
                    pickle.dump(self.images, f)

        print('Total number of images {}'.format(len(self.images)))

    def __getitem__(self, item):
        return self.transform2(self.decompress(self.images[item]))

    def __len__(self):
        return len(self.images)

    @staticmethod
    def compress(img):
        import torchvision.transforms.functional as F
        img_p = F.to_pil_image(img)
        import io
        output = io.BytesIO()
        img_p.save(output, 'JPEG')
        return output

    @staticmethod
    def decompress(output):
        output.seek(0)
        return PIL.Image.open(output)


class SingleImagesFolderMTDatasetWrap(torch.utils.data.Dataset):
    def __init__(self, ds, cache, num_images=None, transform=None):
        self.transform = transform
        if cache is not None and os.path.exists(cache):
            with open(cache, 'rb') as f:
                self.images = pickle.load(f)
        else:
            if num_images is None:
                num_images = len(ds)

            self.images = []
            for i in range(num_images):
                if i % 100 == 0:
                    print(i)
                self.images.append(ds[i][0])

            if cache is not None:
                with open(cache, 'wb') as f:
                    pickle.dump(self.images, f)

        print('Total number of images {}'.format(len(self.images)))

    def __getitem__(self, item):
        if self.transform is not None:
            return self.transform(self.images[item])
        else:
            return self.images[item]

    def __len__(self):
        return len(self.images)

class SingleImagesFolderMTDataset_Index(torch.utils.data.Dataset):
    def __init__(self, root, cache, transform=None, workers=8, split_size=200, protocol=None, num_images=None):
        if cache is not None and os.path.exists(cache):
            with open(cache, 'rb') as f:
                self.images = pickle.load(f)[:39900]
        else:
            self.transform = transform if not transform is None else lambda x: x
            self.images = []

            def split_seq(seq, size):
                newseq = []
                splitsize = 1.0 / size * len(seq)
                for i in range(size):
                    newseq.append(seq[int(round(i * splitsize)):int(round((i + 1) * splitsize))])
                return newseq

            def map(path_imgs):
                imgs_0 = [self.transform(np.array(PIL.Image.open(os.path.join(root, p_i)))) for p_i in path_imgs]
                imgs_1 = [self.compress(img) for img in imgs_0]

                print('.')
                return imgs_1

            path_imgs = os.listdir(root)
            if num_images:
                path_imgs = path_imgs[:num_images]

            n_splits = len(path_imgs) // split_size
            path_imgs_splits = split_seq(path_imgs, n_splits)

            from multiprocessing.dummy import Pool as ThreadPool
            pool = ThreadPool(workers)
            results = pool.map(map, path_imgs_splits)
            pool.close()
            pool.join()

            for r in results:
                self.images.extend(r)

            if cache is not None:
                with open(cache, 'wb') as f:
                    pickle.dump(self.images, f, protocol=protocol)

        print('Total number of images {}'.format(len(self.images)))

    def __getitem__(self, item):
        return self.decompress(self.images[item]), item

    def __len__(self):
        return len(self.images)

    @staticmethod
    def compress(img):
        return img

    @staticmethod
    def decompress(output):
        return output

class IndexWrapDataset(torch.utils.data.Dataset):
    def __init__(self, orig):
        self.orig = orig

    def __getitem__(self, index):
        return self.orig[index][0], index

    def __len__(self):
        return len(self.orig)

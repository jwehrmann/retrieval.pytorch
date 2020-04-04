import os
import torch
import pickle
import numpy as np
from PIL import Image
from addict import Dict
from pathlib import Path
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader

from ..utils.logger import get_logger
from ..utils.file_utils import read_txt, load_pickle
from .preprocessing import get_transform

logger = get_logger()


class Birds(Dataset):
    def __init__(self, data_path, data_name, transform=None,
                target_transform=None, data_split='train',
                tokenizer=None, lang=None):

        self.data_path = data_path
        self.data_name = data_name
        self.tokenizer = tokenizer
        self.target_transform = target_transform
        self.data_split = data_split
        self.__dataset_path = Path(self.data_path) / self.data_name / data_split

        self.embeddings = load_pickle(
            path=os.path.join(self.__dataset_path, 'char-CNN-RNN-embeddings.pickle'),
            encoding='latin1'
        )

        self.fnames = load_pickle(
            path=os.path.join(self.__dataset_path, 'filenames.pickle'),
            encoding='latin1'
        )

        self.images = load_pickle(
            path=os.path.join(self.__dataset_path, '304images.pickle'),
            encoding='latin1'
        )

        self.class_info = load_pickle(
            path=os.path.join(self.__dataset_path, 'class_info.pickle'),
            encoding='latin1'
        )

        self.captions = self._get_captions()
        self.transform = get_transform(data_split)

        self.captions_per_image = 10

        logger.info(f'[Birds] #Captions {len(self.captions)}')
        logger.info(f'[Birds] #Images {len(self.images)}')
        self.n = self._set_len_captions(data_split)

    def __getitem__(self, ix):
        img = Image.fromarray(self.images[ix//self.captions_per_image]).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        fname = self.fnames[ix//self.captions_per_image]
        caption = self.captions[ix]
        tokens = self.tokenizer(caption)
        return img, tokens, ix, fname

    def __len__(self):
        return self.n

    def __repr__(self):
        return f'Birds.{self.data_name}.{self.data_split}'

    def __str__(self):
        return f'{self.data_name}.{self.data_split}'

    def _get_captions(self):
        captions = []
        for fname in self.fnames:
            cap_file = Path(self.data_path) / self.data_name / 'text_c10' / f'{fname}.txt'
            with open(cap_file, 'r') as f:
                cap = f.readlines()
                captions.extend(cap)
        return captions

    def _set_len_captions(self, data_split):
        n = len(self.captions)
        if data_split in ['test', 'dev']:
            n = 10000
        return n


class PrecompDataset(Dataset):
    """
    Load precomputed captions and image features
    Possible options: f30k_precomp, coco_precomp
    """

    def __init__(
        self, data_path, data_name,
        data_split, tokenizers, lang='en',
    ):
        logger.debug(f'Precomp dataset\n {[data_path, data_split, tokenizers, lang]}')
        self.tokenizers = tokenizers
        self.lang = lang
        self.data_split = '.'.join([data_split, lang])
        self.data_path = data_path = Path(data_path)
        self.data_name = Path(data_name)
        self.full_path = self.data_path / self.data_name
        # Load Captions
        caption_file = self.full_path / f'{data_split}_caps.{lang}.txt'
        self.captions = read_txt(caption_file)
        logger.debug(f'Read captions. Found: {len(self.captions)}')

        # Load Image features
        img_features_file = self.full_path / f'{data_split}_ims.npy'
        self.images = np.load(img_features_file)
        self.length = len(self.captions)
        self.ids = np.loadtxt(data_path/ data_name / f'{data_split}_ids.txt', dtype=int)

        self.captions_per_image = 5

        logger.debug(f'Read feature file. Shape: {len(self.images.shape)}')

        # Each image must have five captions
        assert (
            self.images.shape[0] == len(self.captions)
            or self.images.shape[0]*5 == len(self.captions)
        )

        if self.images.shape[0] != len(self.captions):
            self.im_div = 5
        else:
            self.im_div = 1
        # the development set for coco is large and so validation would be slow
        # if data_split == 'dev' and self.length > 5000:
        #     self.length = 5000

        print('Image div', self.im_div)

        logger.info('Precomputing captions')
        # self.precomp_captions =  [
        #     self.tokenizer(x)
        #     for x in self.captions
        # ]

        # self.maxlen = max([len(x) for x in self.precomp_captions])
        # logger.info(f'Maxlen {self.maxlen}')

        logger.info((
            f'Loaded PrecompDataset {self.data_name}/{self.data_split} with '
            f'images: {self.images.shape} and captions: {self.length}.'
        ))

    def get_img_dim(self):
        return self.images.shape[-1]

    def __getitem__(self, index):
        # handle the image redundancy
        img_id = index//self.im_div
        image = self.images[img_id]
        image = torch.FloatTensor(image)

        # caption = self.precomp_captions[index]
        caption = self.captions[index]

        ret_caption = []
        for tokenizer in self.tokenizers:
            tokens = tokenizer(caption)
            ret_caption.append(tokens)

        batch = Dict(
            image=image,
            caption=ret_caption,
            index=index,
            img_id=img_id,
        )

        return batch

    def __len__(self):
        return self.length

    def __repr__(self):
        return f'PrecompDataset.{self.data_name}.{self.data_split}'

    def __str__(self):
        return f'{self.data_name}.{self.data_split}'


class DummyDataset(Dataset):
    """
    Load precomputed captions and image features
    Possible options: f30k_precomp, coco_precomp
    """

    def __init__(
        self, data_path, data_name,
        data_split, tokenizer, lang='en'
    ):
        logger.debug(f'Precomp dataset\n {[data_path, data_split, tokenizer, lang]}')
        self.tokenizer = tokenizer

        self.captions = np.random.randint(0, 1000, size=(5000, 50))
        logger.debug(f'Read captions. Found: {len(self.captions)}')

        # Load Image features
        self.images = np.random.uniform(size=(1000, 36, 2048))
        self.length = 5000

        logger.debug(f'Read feature file. Shape: {len(self.images.shape)}')

        # Each image must have five captions
        assert (
            self.images.shape[0] == len(self.captions)
            or self.images.shape[0]*5 == len(self.captions)
        )

        if self.images.shape[0] != len(self.captions):
            self.im_div = 5
        else:
            self.im_div = 1
        # the development set for coco is large and so validation would be slow
        if data_split == 'dev':
            self.length = 5000
        print('Image div', self.im_div)

        # self.precomp_captions =  [
        #     self.tokenizer(x)
        #     for x in self.captions
        # ]

        # self.maxlen = max([len(x) for x in self.precomp_captions])
        # logger.info(f'Maxlen {self.maxlen}')

    def get_img_dim(self):
        return self.images.shape[-1]

    def __getitem__(self, index):
        # handle the image redundancy
        img_id = index//self.im_div
        image = self.images[img_id]
        image = torch.FloatTensor(image)
        # caption = self.precomp_captions[index]
        caption = torch.LongTensor(self.captions[index])

        return image, caption, index, img_id

    def __len__(self):
        return self.length


class CrossLanguageLoader(Dataset):
    """
    Load precomputed captions and image features
    Possible options: f30k_precomp, coco_precomp
    """

    def __init__(
        self, data_path, data_name, data_split,
        tokenizers, lang='en-de',
    ):
        logger.debug((
            'CrossLanguageLoader dataset\n '
            f'{[data_path, data_split, tokenizers, lang]}'
        ))

        self.data_path = Path(data_path)
        self.data_name = Path(data_name)
        self.full_path = self.data_path / self.data_name
        self.data_split = '.'.join([data_split, lang])

        self.lang = lang

        assert len(tokenizers) == 1 # TODO: implement multi-tokenizer

        self.tokenizer = tokenizers[0]

        lang_base, lang_target = lang.split('-')
        base_filename = f'{data_split}_caps.{lang_base}.txt'
        target_filename = f'{data_split}_caps.{lang_target}.txt'

        base_file = self.full_path / base_filename
        target_file = self.full_path / target_filename

        logger.debug(f'Base: {base_file} - Target: {target_file}')
        # Paired files
        self.lang_a = read_txt(base_file)
        self.lang_b = read_txt(target_file)

        logger.debug(f'Base and target size: {(len(self.lang_a), len(self.lang_b))}')
        self.length = len(self.lang_a)
        assert len(self.lang_a) == len(self.lang_b)

        logger.info((
            f'Loaded CrossLangDataset {self.data_name}/{self.data_split} with '
            f'captions: {self.length}'
        ))

    def __getitem__(self, index):
        caption_a = self.lang_a[index]
        caption_b = self.lang_b[index]

        target_a = self.tokenizer(caption_a)
        target_b = self.tokenizer(caption_b)

        return target_a, target_b, index

    def __len__(self):
        return self.length

    def __str__(self):
        return f'{self.data_name}.{self.data_split}'


class ImageDataset(Dataset):
    """
    Load precomputed captions and image features
    Possible options: f30k_precomp, coco_precomp
    """

    def __init__(
        self, data_path, data_name,
        data_split, tokenizer, lang='en',
        resize_to=256, crop_size=224,
    ):
        from .adapters import Flickr, Coco

        logger.debug(f'ImageDataset\n {[data_path, data_split, tokenizer, lang]}')
        self.tokenizer = tokenizer
        self.lang = lang
        self.data_split = data_split
        self.split = '.'.join([data_split, lang])
        self.data_path = Path(data_path)
        self.data_name = Path(data_name)
        self.full_path = self.data_path / self.data_name

        self.data_wrapper = (
            Flickr(
                self.full_path,
                data_split=data_split,
            ) if 'f30k' in data_name
            else Coco(
                self.full_path,
                data_split=data_split,
            )
        )

        self._fetch_captions()
        self.length = len(self.ids)

        self.transform = get_transform(
            data_split, resize_to=resize_to, crop_size=crop_size
        )

        self.captions_per_image = 5

        if data_split == 'dev' and len(self.length) > 5000:
            self.length = 5000

        logger.debug(f'Split size: {len(self.ids)}')

    def _fetch_captions(self,):
        self.captions = []
        for image_id in sorted(self.data_wrapper.image_ids):
            self.captions.extend(
                self.data_wrapper.get_captions_by_image_id(image_id)[:5]
            )

        self.ids = range(len(self.captions))
        logger.debug(f'Loaded {len(self.captions)} captions')

    def load_img(self, image_id):

        filename = self.data_wrapper.get_filename_by_image_id(image_id)
        feat_path = self.full_path / filename
        try:
            image = default_loader(feat_path)
            image = self.transform(image)
        except OSError:
            print('Error to load image: ', feat_path)
            image = torch.zeros(3, 224, 224,)

        return image

    def __getitem__(self, index):
        # handle the image redundancy
        seq_id = self.ids[index]
        image_id = self.data_wrapper.image_ids[seq_id//5]

        image = self.load_img(image_id)

        caption = self.captions[index]
        cap_tokens = self.tokenizer(caption)

        return image, cap_tokens, index, image_id

    def __len__(self):
        return self.length

    def __repr__(self):
        return f'ImageDataset.{self.data_name}.{self.split}'

    def __str__(self):
        return f'{self.data_name}.{self.split}'

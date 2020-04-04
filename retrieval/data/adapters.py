from ..utils.file_utils import load_json
from collections import defaultdict
from ..utils.logger import get_logger
from pathlib import Path


logger = get_logger()


class Flickr:

    def __init__(self, data_path, data_split):

        data_split = data_split.replace('dev', 'val')

        self.data_path = Path(data_path)
        self.annotation_path = (
            self.data_path / 'dataset_flickr30k.json'
        )
        self.data = load_json(self.annotation_path)
        self.image_ids, self.img_dict, self.img_captions = self._get_img_ids(data_split)

        for k, v in self.img_captions.items():
            assert len(v) == 5

        logger.info((
            f'[Flickr] Loaded {len(self.img_captions)} images '
            f'and {len(self.img_captions)*5} annotations.'
        ))

    def _get_img_ids(self, data_split):
        image_ids = []
        img_dict = {}
        annotations = defaultdict(list)
        for img in self.data['images']:
            if img['split'].lower() != data_split.lower():
                continue
            img_dict[img['imgid']] = img
            image_ids.append(img['imgid'])
            annotations[img['imgid']].extend(
                [x['raw'] for x in img['sentences']][:5]
            )
        return image_ids, img_dict, annotations

    def get_image_id_by_filename(self, filename):
        return self.img_dict[filename]['imgid']

    def get_captions_by_image_id(self, img_id):
        return self.img_captions[img_id]

    def get_filename_by_image_id(self, image_id):
        return (
            Path('images') /
            Path('flickr30k_images') /
            self.img_dict[image_id]['filename']
        )

    def __call__(self, filename):
        return self.img_dict[filename]

    def __len__(self, ):
        return len(self.img_captions)


class Coco:

    def __init__(self, path, data_split):

        data_split = data_split.replace('dev', 'val')

        self.data_path = Path(path)
        self.annotation_path = (
            self.data_path / 'dataset_coco.json'
        )
        self.data = load_json(self.annotation_path)

        self.image_ids, self.img_dict, self.img_captions = self._get_img_ids(data_split)

        for k, v in self.img_captions.items():
            assert len(v) == 5

        logger.info((
            f'[Coco] Loaded {len(self.img_captions)} images '
            f'and {len(self.img_captions)*5} annotations.'
        ))

    def _get_img_ids(self, data_split):
        img_dict = {}
        image_ids = []
        annotations = defaultdict(list)
        for img in self.data['images']:
            split = img['split'].lower().replace('restval', 'train')
            if split != data_split.lower():
                continue
            img_dict[img['imgid']] = img
            image_ids.append(img['imgid'])

            annotations[img['imgid']].extend(
                [x['raw'] for x in img['sentences']][:5]
            )
        return image_ids, img_dict, annotations

    def get_image_id_by_filename(self, filename):
        return self.img_dict[filename]['imgid']

    def get_captions_by_image_id(self, img_id):
        return self.img_captions[img_id]

    def get_filename_by_image_id(self, image_id):
        return (
            Path('images') /
            self.img_dict[image_id]['filename'].split('_')[1] /
            self.img_dict[image_id]['filename']
        )

    def __call__(self, filename):
        return self.img_dict[filename]

    def __len__(self, ):
        return len(self.img_captions)

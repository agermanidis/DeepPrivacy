from deep_privacy.inference import deep_privacy_anonymizer, infer
from deep_privacy.detection import detection_api
from deep_privacy import config_parser, utils

import runway
import numpy as np
import shutil

@runway.setup(options={'checkpoint_dir': runway.file(is_directory=True), 'face_detector': runway.file(extension='.pth')})
def setup(opts):
    shutil.move(opts['face_detector'], 'deep_privacy/detection/dsfd/weights/WIDERFace_DSFD_RES152.pth')
    config = config_parser.load_config('models/default/config.yml')
    ckpt = utils.load_checkpoint(opts['checkpoint_dir'])
    generator = infer.init_generator(config, ckpt)
    anonymizer = deep_privacy_anonymizer.DeepPrivacyAnonymizer(generator, 128, use_static_z=True)
    return anonymizer

@runway.command('anonymize', inputs={'image': runway.image}, outputs={'anonymized_image': runway.image})
def anonymize(anonymizer, inputs):
   images = [np.array(inputs['image'])]
   im_bboxes, im_keypoints = detection_api.batch_detect_faces_with_keypoints(
            images, im_bboxes=None,
            keypoint_threshold=anonymizer.keypoint_threshold,
            face_threshold=anonymizer.face_threshold
        )
   anonymized_images = anonymizer.anonymize_images(images, im_keypoints=im_keypoints, im_bboxes=im_bboxes)
   return anonymized_images[0]

if __name__ == '__main__':
   runway.run()
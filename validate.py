"""
Inference Yolov4 with Triton. Heads model.
"""

import argparse
import pathlib

import cv2
from tritonclient.utils import *
import tritonclient.grpc as grpcclient

from utils import calc_darknet_map, read_darknet_result_json, get_all_darknet_anno

BATCH_SIZE = 1


def parse_model_grpc(model_metadata, model_config):
    """
    Check the configuration of a model to make sure it meets the
    requirements for an image classification network (as expected by
    this client)
    """
    if len(model_metadata.inputs) != 1:
        raise Exception("expecting 1 input, got {}".format(
            len(model_metadata.inputs)))

    if len(model_config.config.input) != 1:
        raise Exception(
            "expecting 1 input in model configuration, got {}".format(
                len(model_config.config.input)))

    input_metadata = model_metadata.inputs[0]
    outputs_metadata = model_metadata.outputs
    input_config = model_config.config.input[0]

    max_batch_size = 0
    if model_config.config.max_batch_size:
        max_batch_size = model_config.config.max_batch_size
    return (input_metadata.name, outputs_metadata, max_batch_size)


class ModelMeta:
    def __init__(self, **entries):
        self.__dict__.update(entries)
        self.d = entries

    def __getitem__(self, item):
        return self.d[item]


class YoloNet:
    def __init__(self, is_http: bool,
                 url: str,
                 model_name: str,
                 model_version: str = '1'
                 ):
        self.client_type = None
        self.url = url
        self.model_name = model_name
        self.model_version = model_version

        self.triton_init(is_http)

    def triton_init(self, is_http: bool):
        try:
            self.client = grpcclient
            self.inf_server_client = grpcclient.InferenceServerClient(url=self.url, verbose=False)
        except Exception as e:
            print("client creation failed: " + str(e))
            raise
        try:
            self.model_metadata = self.inf_server_client.get_model_metadata(model_name=self.model_name,
                                                                            model_version=self.model_version
                                                                            )
            self.model_config = self.inf_server_client.get_model_config(model_name=self.model_name,
                                                                        model_version=self.model_version
                                                                        )
        except InferenceServerException as e:
            print(f"failed to retrieve the metadata and config: {str(e)}")
            raise
        except ConnectionRefusedError as e:
            print(f"Connection refused: {str(e)}")
            raise

        try:
            print('is_server_live:', self.inf_server_client.is_server_live())
            print('is_server_ready:', self.inf_server_client.is_server_ready())
            print('models_list:', str(self.inf_server_client.get_model_repository_index()).replace('}, {', '},\n{'))
            # print('statistics:', self.inf_server_client.get_inference_statistics(model_name))
            # print('unload_model:', 'ok' if self.inf_server_client.unload_model(model_name) is None else '') #Only with --model-control-mode=explicit
            # print('load_model:', 'ok' if self.inf_server_client.load_model(model_name) is None else '')
            print('is_model_ready:', self.inf_server_client.is_model_ready(self.model_name, self.model_version))
        except ConnectionRefusedError as e:
            raise

    def get_metadata(self):
        # Get model metadata from triton
        input_name, output_metadata, batch_size = parse_model_grpc(self.model_metadata, self.model_config)
        input_type = self.model_metadata.inputs[0].datatype
        output_names = [output.name for output in output_metadata]
        out = {'input_name': input_name, 'input_type': input_type, 'output_names': output_names,
               'batch_size': batch_size}
        return ModelMeta(**out)

    def get_metadata2(self):
        # Get model metadata from triton
        input_name, output_metadata, batch_size = parse_model_grpc(self.model_metadata, self.model_config)
        input_type = self.model_metadata['inputs'][0]['datatype']
        output_names = [output['name'] for output in output_metadata]
        out = {'input_name': input_name, 'input_type': input_type, 'output_names': output_names,
               'batch_size': batch_size}
        return ModelMeta(**out)

    def inference_client2(self, batched_images, model_meta):
        input0_data, input0_shape = batched_images
        out = {}
        try:
            with self.client.InferenceServerClient(self.url, verbose=False) as client:
                inputs = [self.client.InferInput(model_meta.input_name, input0_shape, model_meta.input_type), ]
                inputs[0].set_data_from_numpy(input0_data)
                outputs = []
                for output_name in model_meta.output_names:
                    outputs.append(self.client.InferRequestedOutput(output_name))
                response = client.infer(self.model_name, inputs, request_id=str(1), outputs=outputs)
                result = response.get_response()
            for r in result.outputs:
                out[r.name] = response.as_numpy(r.name)
        except ConnectionRefusedError as e:
            print('Error: {}'.format(e))
            raise

        return out

    def detect(self, imgs):
        imgs = imgs[0]
        batch_imgs = np.array(imgs), np.shape(imgs)
        model_meta = self.get_metadata()
        res = self.inference_client2(batch_imgs, model_meta)
        result = []
        for i in range(len(imgs)):
            result.append({
                "boxes": res['OUTPUT_01_BOXES'],
                "classes": res['OUTPUT_01_CLASS'],
                "probs": res['OUTPUT_01_PROBS']
            })
        return result


def tlbr_to_darknet_relative(bbox, img_size):
    # bbox [[x1,y1,x2,y2]
    # img_size = (batch, height, width, ch)
    _, h, w, _ = img_size
    x1 = bbox[:, 0]
    y1 = bbox[:, 1]
    x2 = bbox[:, 2]
    y2 = bbox[:, 3]
    center_box = np.stack([(x1 + x2) / 2,
                            (y1 + y2) / 2,
                            (x2 - x1),
                            (y2 - y1)
                            ]).transpose()

    center_box[:, [0, 2]] /= w
    center_box[:, [1, 3]] /= h
    return center_box


def darknet_relative_to_tlbr(center_box, img_size):

    # bbox [[center_x,center_y,width,height]
    # img_size = (batch, height, width, ch)
    _, h, w, _ = img_size
    center_box[:, [0, 2]] *= w
    center_box[:, [1, 3]] *= h
    x_center = center_box[:, 0]
    y_center = center_box[:, 1]
    width = center_box[:, 2]
    height = center_box[:, 3]
    center_box = np.stack([(x_center - width / 2),
                            (y_center - height / 2),
                            (x_center + width / 2),
                            (y_center + height / 2)
                            ]).transpose()

    center_box[center_box < 0] = 0
    return center_box.astype(np.int)


def get_files(root_dir, extensions):
    all_files = []
    for ext in extensions:
        all_files.extend(root_dir.glob(ext))
    return all_files


def run(opt):
    # load model
    model = YoloNet(is_http=False,
                    url=opt.url,
                    model_name=opt.model,
                    model_version='1'
                    )
    darknet_result = read_darknet_result_json(opt.json_input)

    images_dir = opt.images_dir if opt.images_dir != pathlib.Path('no.dir') else None
    if images_dir is not None:
        files = get_files(images_dir, ('*.jpg', '*.jpeg', '*.png'))
    else:
        files = list(darknet_result.keys())
    detections = {}
    for file in files:
        frame = cv2.imread(str(file))
        frame = frame.copy()[np.newaxis, ...]

        # model predict
        output = model.detect(np.stack([frame]))

        for idx, out in enumerate(output):
            if out["boxes"].size != 0:
                # bbox [[x1,y1,x2,y2]
                bbox = out["boxes"][0].astype('float32')
                probs = out["probs"][0].reshape(-1, 1).astype('float32')
                classes = out["classes"][0].reshape(-1, 1).astype('float32')
                rel_boxes = tlbr_to_darknet_relative(bbox, frame.shape)
                detections[file] = np.hstack([classes, rel_boxes, probs])
            else:
                detections[file] = np.asarray([])

    anno = get_all_darknet_anno(files)
    print(f'Darknet model result: {calc_darknet_map(anno, darknet_result)}')
    print(f'TRT model result: {calc_darknet_map(anno, detections)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', type=str,
                        default='0.0.0.0:8001', help='TritonServer ip:port')
    parser.add_argument('--model', type=str,
                        default='yolov4_1_class_ensemble_nodec', help='TRT model name in TritonServer')
    parser.add_argument('--images_dir', type=pathlib.Path,
                        default='no.dir', help='Directory with images')
    parser.add_argument('--output', type=str, default='',
                        help='output folder')  # output folder
    parser.add_argument('--conf-thres', type=float,
                        default=0.5, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float,
                        default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--json_input', type=pathlib.Path, default='example/result.json',
                        help='output video codec (verify ffmpeg support)')

    args = parser.parse_args()
    run(args)

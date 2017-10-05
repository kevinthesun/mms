import mms.utils.mxnet.image as image
import mms.utils.mxnet.ndarray as ndarray

from mms.model_service.mxnet_model_service import MXNetBaseService


class MXNetVisionService(MXNetBaseService):
    def _preprocess(self, data):
        img_list = []
        for idx, img in enumerate(data):
            input_shape = self.signature['inputs'][idx]['data_shape']
            # We are assuming input shape is NCHW
            [h, w] = input_shape[2:]
            img_arr = image.read(img)
            img_arr = image.resize(img_arr, w, h)
            img_arr = image.transform_shape(img_arr)
            img_list.append(img_arr)
        return img_list

    def _postprocess(self, data):
        return [ndarray.top_probability(d, self.labels, top=5) for d in data]


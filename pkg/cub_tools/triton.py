
from tritonclient import http

def run_inference(X, X_shape=(1, 3, 224,  224), X_dtype='FP32', model_name='cub200_resnet34', input_name=['INPUT__0'], output_name='OUTPUT__0',
                  url='ecm-clearml-compute-gpu-002.westeurope.cloudapp.azure.com', model_version='1', port=8000, VERBOSE=False):
    url = url+':'+str(port)
    triton_client = http.InferenceServerClient(url=url, verbose=VERBOSE)
  
    input0 = http.InferInput(input_name[0], X_shape, X_dtype)
    input0.set_data_from_numpy(X, binary_data=False)
    output = http.InferRequestedOutput(output_name,  binary_data=False)
    response = triton_client.infer(model_name, model_version=model_version, inputs=[input0], outputs=[output])
    y_pred_proba = response.as_numpy(output_name)
    y_pred = y_pred_proba.argmax(1)

    return y_pred_proba, y_pred


def get_model_info(model_name='cub200_resnet34', model_version='1', url='ecm-clearml-compute-gpu-002.westeurope.cloudapp.azure.com',port=8000, VERBOSE=False):
    url = url+':'+str(port)
    triton_client = http.InferenceServerClient(url=url, verbose=VERBOSE)
    model_metadata = triton_client.get_model_metadata(model_name=model_name, model_version=model_version)
    model_config = triton_client.get_model_config(model_name=model_name, model_version=model_version)

    return model_config, model_metadata
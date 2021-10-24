import device.exp_device
import device.opportunistic_device
import device.hetero_device
import device.tmc_exp_device
import device.federated_device
import device.quantization_device
import device.droppcl_device

def get_device_class(class_name):
    ### Hetero & Dropout
    if class_name == 'hetero':
        return device.hetero_device.HeteroDevice
    elif class_name == 'dropin':
        return device.hetero_device.DropinDevice
    elif class_name == 'dropinnout':
        return device.hetero_device.DropInNOutDevice
    elif class_name == 'dropout only':
        return device.hetero_device.DropoutOnlyOnDevice
    elif class_name == 'mixed dropout':
        return device.hetero_device.MixedDropoutDevice
    elif class_name == 'mixed dropout 2':
        return device.hetero_device.MixedDropoutDevice
    elif class_name == 'dyn mixed dropout':
        return device.hetero_device.DynamicMixedDropoutDevice
    elif class_name == 'no dropout':
        return device.hetero_device.NoDropoutDevice
    elif class_name == 'mixed scaled dropout':
        return device.hetero_device.MixedScaledDropoutDevice
    elif class_name == 'mixed multiopt dropout':
        return device.hetero_device.MixedMultiOptDropoutDevice 
    elif class_name == 'momentum dropout':
        return device.hetero_device.MomentumMixedDropoutDevice
    elif class_name == 'auto m. dropout':
        return device.hetero_device.AutoMomentumMixedDropoutDevice

    ### Quantization
    elif class_name == 'Q. grad':
        return device.quantization_device.QuantizationDevice
    elif class_name == 'No Q':
        return device.quantization_device.NoQuantizationDevice
    elif class_name == 'Q. params':
        return device.quantization_device.QuantizationParamDevice
    elif class_name == 'Q. grad & params':
        return device.quantization_device.QuantizationGradParamDevice
    elif class_name == 'mixed Q. grad':
        return device.quantization_device.MixedQuantizationDevice
    elif class_name == 'Q. Net':
        return device.quantization_device.QuantizationNetworksDevice

    ### DROppCL Devices for final exp.
    elif class_name == 'DROppCL test':
        return device.droppcl_device.DROppCLTestDevice
    elif class_name == 'baseline':
        return device.droppcl_device.DROppCLBaselineDevice
    elif class_name == 'dropout':
        return device.droppcl_device.DROppCLOnlyDropoutDevice
    elif class_name == 'quantize':
        return device.droppcl_device.DROppCLOnlyQuantizationDevice
    elif class_name == 'DROppCL':
        return device.droppcl_device.DROppCLDevice

    ### DROppCL Devices for controlled exp.
    elif class_name == 'c_DROppCL':
        return device.quantization_device.MomentumDROppCLDevice
    elif class_name == 'c_DROppCL Auto':
        return device.quantization_device.AutoMomentumDROppCLDevice
    elif class_name == 'c_only dropout':
        return device.quantization_device.OnlyDropoutDevice
    elif class_name == 'c_only quant':
        return device.quantization_device.OnlyQuantDevice
    elif class_name == 'no dropout nor Q.':
        return device.quantization_device.NoDropoutNorQDevice
        
    #####################
    if class_name == 'greedy':
        client_class = device.exp_device.GreedyWOSimDevice
    elif class_name == 'local':
        client_class = device.exp_device.LocalDevice
    elif class_name == 'opportunistic':
        client_class = device.opportunistic_device.JSDOppDevice
    elif class_name == 'opportunistic-weighted':
        client_class = device.tmc_exp_device.JSDOppWeightedDevice
    elif class_name == 'opportunistic (low thres.)':
        client_class = device.opportunistic_device.LowJSDOppDevice
    elif class_name == 'federated':
        client_class = device.federated_device.FederatedDevice
    elif class_name == 'federated (opportunistic)':
        client_class = device.federated_device.FederatedJSDGreedyDevice
    elif class_name == 'gradient replay':
        client_class = device.gr_device.JSDGradientReplayDevice
    # elif class_name == 'gradient replay no decay':
    #     client_class = JSDGradientReplayNoDecayClient
    # elif class_name == 'gradient replay not weighted':
    #     client_class = JSDGradientReplayNoWeightingClient
    # elif class_name == 'gradient replay decay':
    #     client_class = JSDGradientReplayDecayClient
    # elif class_name == 'gradient replay (low thres.)':
    #     client_class = HighJSDGradientReplayClient
    # # 'cecay': Client-specific dECAY
    # elif class_name == 'greedy-cecay':
    #     client_class = GreedyNoSimCecayClient
    # elif class_name == 'opportunistic-cecay':
    #     client_class = JSDGreedySimCecayClient
    # elif class_name == 'gradient replay cecay':
    #     client_class = JSDGradientReplayCecayClient
    # elif class_name == 'greedy ':
    #     client_class = OnlyOtherGreedyClient
    # elif class_name == 'oracle':
    #     client_class = OracleClient
    # elif class_name == 'task-aware':
    #     client_class = TaskAwareClient
    # elif class_name == 'compressed':
    #     client_class = CompressedTaskAwareClient
    # elif class_name == 'compressed-v2':
    #     client_class = V2CompressedTaskAwareClient
    # elif class_name == 'task-aware GR':
    #     client_class = TaskAwareGradientReplayClient
    # ### CIFAR 100 versions
    # elif class_name == 'oracle ':
    #     client_class = Cifar100OracleClient
    # elif class_name == 'compressed ':
    #     client_class = CompressedCNNTaskAwareClient
    # elif class_name == 'compressed-v2 ':
    #     client_class = V2CompressedCNNTaskAwareClient

    # ### Knowledge Distillation
    # elif class_name == 'only data':
    #     client_class = TrainOnDataClient
    # elif class_name == 'only model':
    #     client_class = TrainOnModelClient
    # elif class_name == 'data model half half':
    #     client_class = TrainOnDataAndModelClient

    # ### clients for checking encounters
    # elif class_name == 'oppo check':
    #     client_class = OpportunisticCheckClient
    # elif class_name == 'oppo local check':
    #     client_class = OpportunisticLocalCheckClient
    # elif class_name == 'oppo check low':
    #     client_class = LowThresOpportunisticCheckClient
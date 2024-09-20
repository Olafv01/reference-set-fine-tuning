import torch
import numpy as np
import math


def SUE(preds, dists,ref_poses, num_NN=20, slope=350):
    sue_scores = np.zeros(len(preds))

    print('Computing SUE uncertainty')    
    weights = np.ones(num_NN)
    for itr in tqdm(range(len(sue_scores))):   
        top_preds = preds[itr][:num_NN]
        nn_poses = ref_poses[top_preds]
        bm_pose = nn_poses[0]

        for itr2 in range(num_NN):
            weights[itr2] = math.e ** ((-1*abs(dists[itr][itr2])) * slope) 

        weights = weights/sum(abs(weights))

        mean_pose = np.asarray([np.average(nn_poses[:,0], weights=weights), np.average(nn_poses[:,1], weights=weights)])

        variance_lat_lat = 0 
        variance_lon_lon = 0    
        variance_lat_lon = 0    

        for k in range(0, num_NN):                
            diff_lat_lat = min(500, nn_poses[k,0] - mean_pose[0]) # so everything that is more than 500 meters away contributes equally to the variance 
            diff_lon_lon = min(500, nn_poses[k,1] - mean_pose[1])
            diff_lat_lon = min(500, nn_poses[k,0] - mean_pose[0]) *  min(500, nn_poses[k,1] - mean_pose[1])

            variance_lat_lat = variance_lat_lat + weights[k] * (diff_lat_lat)**2
            variance_lon_lon = variance_lon_lon + weights[k] * (diff_lon_lon)**2
            variance_lat_lon = variance_lat_lon + weights[k] * diff_lat_lon

        sue_scores[itr] = (variance_lat_lat + variance_lon_lon)/2  # assuming independent dimensions

    # sue_scores = -1 * sue_scores # converting into a confidence instead of an uncertainty
    # sue_scores_normalized = np.interp(sue_scores, (sue_scores.min(), sue_scores.max()), (0.0, 0.9999)) # avoiding infinity
    print('Done!') 
    return sue_scores

def get_model(args,name,pth):
    rerank=None
    if name=="sela":
        import vpr_models.SelaVPR
        from vpr_models import get_model
        args.image_size=[224,224]
        args.resize=args.image_size
        args.features_dim=1024
        args.foundation_model_path=pth
        rerank , model= vpr_models.SelaVPR.get_model(args)
        model.cuda()

        new_state=torch.load(pth)
        if new_state!=None:
            state_dict=new_state
            if "model_state_dict" in state_dict.keys():
                        state_dict=state_dict["model_state_dict"]

            model.load_state_dict(state_dict)
            print("model on {} loaded!".format(pth))
            
        else:
            print("No new state defined")
            
    elif name=="boq":

        from vpr_models import get_model
        if args.backbone is None:
            args.backbone="dinov2"
        if args.backbone.lower() not in [None, "dinov2","resnet50"]:
            raise ValueError(f"When using BOQ the backbone must be None or resnet50 or Dinov2 not {args.backbone}")
        
        if args.backbone.lower() == "dinov2":
            args.descriptors_dimension=12288
            args.image_size=[322,322]
            args.resize=args.image_size
        elif args.backbone.lower() == "resnet50":
            args.descriptors_dimension=16384
            args.image_size=[384,384]
            args.resize=args.image_size
        model= torch.hub.load("amaralibey/bag-of-queries", "get_trained_boq", backbone_name=args.backbone.lower(), output_dim=args.descriptors_dimension)
        
        args.features_dim=args.descriptors_dimension
        

    elif name == "crica":
        args.image_size=[224,224]
        args.resize=args.image_size
        args.descriptors_dimension = 10752
        model=torch.hub.load("Lu-Feng/CricaVPR", "trained_model")
    #     print(sys.path)
    #     sy.pth()
        args.features_dim=args.descriptors_dimension        
        
    return model,rerank

class boq_output_only_model():
    def __init__(self,model):
        self.model=model
        
    def train(self):
        self.model.train()
        return self
        
    def eval(self):
        self.model= self.model.eval()
        return self
        
    def cuda(self):
        self.model.cuda()
        return self
        
    def load_state_dict(self,state):
        self.model.load_state_dict(state)
        return self
        
    def parameters(self):
        return self.model.parameters()
    
    def state_dict(self):
        return self.model.state_dict()
    
    def output_only(self,imgs):
        output,_ = self.model(imgs)
        return output
    
    def __call__(self,imgs):
        output,_=self.model(imgs)
        return output
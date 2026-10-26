from assoc_common import *
def main():
 verify();bundle=BASE_ROOT/'bundle-seed0';meta=read(bundle/'bundle.json')
 for name,h in meta['files_sha256'].items():assert sha(bundle/name)==h
 model=Student();backbone=SharedBackbone(p.WEIGHTS);det=torch.load(bundle/'yolo.pt',map_location='cpu',weights_only=False)['model'];rule=read(ROOT/'rules.json');single=np.array(rule['weights']);pair=np.array(rule['pair_weights']);save(ROOT/'model-inventory.json',dict(time=time.time(),student_head_parameters=sum(x.numel() for x in model.parameters()),student_encoder_parameters=sum(x.numel() for x in backbone.parameters()),semantic_total_parameters=sum(x.numel() for x in model.parameters())+sum(x.numel() for x in backbone.parameters()),detector_parameters=sum(x.numel() for x in det.parameters()),rule_dense_coefficients=int(single.size+pair.size),rule_nonzero_coefficients=int((single!=0).sum()+(pair!=0).sum()),rule_trainable_parameters=0,rule_manifest_bytes=(ROOT/'rules.json').stat().st_size,bundle_manifest_sha256=sha(bundle/'bundle.json'),source='Actual unchanged bundle/model weights; CPU counts only. Postprocessing coefficients are separate nontrainable constants, not student network parameters.'))
 print(read(ROOT/'model-inventory.json'))
if __name__=='__main__':main()

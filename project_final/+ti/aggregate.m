function [r,probs] = aggregate(m,p,labels)
ids=unique(m.recording_id,'stable'); probs=zeros(numel(ids),numel(labels));
truth=zeros(numel(ids),1); votes=zeros(numel(ids),1);
[~,segmentIdx]=max(p,[],2); segmentLabels=labels(segmentIdx);
for j=1:numel(ids)
    use=m.recording_id==ids(j); y=unique(m.instrument_id(use));
    assert(numel(y)==1,'Mixed source labels.'); truth(j)=y;
    probs(j,:)=mean(p(use,:),1); votes(j)=mode(segmentLabels(use));
end
[~,idx]=max(probs,[],2);
r=table(ids,truth,labels(idx),votes,'VariableNames', ...
    {'recording_id','instrument_id','prediction','majority_prediction'});
end

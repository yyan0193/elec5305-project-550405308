function [labels,prob] = predict(model,X)
Z=(double(X)-model.mu)./model.sigma; prob=zeros(size(Z,1),numel(model.classes));
for c=1:numel(model.classes)
    [~,score]=predict(model.binary{c},Z);
    positive=find(model.binary{c}.ClassNames==true);
    prob(:,c)=score(:,positive);
end
prob=max(prob,realmin); prob=prob./sum(prob,2);
[~,idx]=max(prob,[],2); labels=model.classes(idx);
assert(all(isfinite(prob(:))),'Nonfinite model scores.');
end

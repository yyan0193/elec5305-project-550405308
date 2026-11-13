function model = train(X,y,recordingIDs,cfg)
% Same MATLAB one-vs-rest ridge logistic classifier for both representations.
% This differs from Stage 2's sklearn multinomial logistic regression.
X=double(X); model.mu=mean(X,1); model.sigma=std(X,1,1);
model.sigma(model.sigma<1e-12)=1;
Z=(X-model.mu)./model.sigma; w=ti.weights(recordingIDs);
assert(all(isfinite(Z(:))),'Invalid training matrix.');
assert(isequal(sort(unique(y)),cfg.labels),'Training must contain all classes.');
model.classes=cfg.labels; model.binary=cell(11,1);
for c=1:11
    target=y==cfg.labels(c);
    prior=[sum(w(~target)),sum(w(target))]/sum(w);
    model.binary{c}=fitclinear(Z,target,'Learner','logistic','Regularization','ridge', ...
        'Lambda',1/(cfg.C*sum(w)),'Solver','lbfgs','Weights',w, ...
        'ClassNames',[false;true],'Prior',prior,'GradientTolerance',1e-6);
end
model.classifier='MATLAB one-vs-rest L2 logistic; row-normalised positive scores';
end

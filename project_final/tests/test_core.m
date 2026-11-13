function tests = test_core
tests=functiontests(localfunctions);
end

function testManifest(t)
cfg=ti.config(); m=ti.manifest(cfg);
verifyEqual(t,height(m),988); verifyEqual(t,numel(unique(m.recording_id)),55);
end

function testBudgetCounts(t)
counts=zeros(1,4);
for f=1:5
    remain=setdiff(1:5,f);
    for b=1:4, counts(b)=counts(b)+size(nchoosek(remain,b),1); end
end
verifyEqual(t,counts,[20 30 20 5]); verifyEqual(t,2*sum(counts),150);
end

function testWeights(t)
ids=["a";"a";"b";"c";"c";"c"]; w=ti.weights(ids);
verifyEqual(t,mean(w),1,'AbsTol',1e-12);
verifyEqual(t,sum(w(ids=="a")),sum(w(ids=="b")),'AbsTol',1e-12);
verifyEqual(t,sum(w(ids=="c")),sum(w(ids=="b")),'AbsTol',1e-12);
end

function testMetrics(t)
s=ti.metrics([1;1;2;3],[1;2;2;2],(1:3)');
verifyEqual(t,s.accuracy,.5); verifyEqual(t,s.macro_f1,7/18,'AbsTol',1e-12);
verifyEqual(t,sum(s.confusion(:)),4);
end

function testAggregation(t)
m=table(["a";"a";"b"],[1;1;2],'VariableNames',{'recording_id','instrument_id'});
p=[.9 .1;.2 .8;.1 .9]; [r,q]=ti.aggregate(m,p,[1;2]);
verifyEqual(t,r.prediction,[1;2]); verifyEqual(t,q(1,:),[.55 .45],'AbsTol',1e-12);
end

function testFeatures(t)
cfg=ti.config(); y=sin(2*pi*440*(0:119999)'/24000);
[x,names,frames]=ti.features(y,cfg);
verifySize(t,x,[1 64]); verifyEqual(t,numel(names),64); verifySize(t,frames,[32 231]);
verifyTrue(t,all(isfinite(x))); verifyTrue(t,x(names=="spectral_centroid_mean")>400);
verifyTrue(t,x(names=="spectral_centroid_mean")<500);
end

function testSilence(t)
cfg=ti.config(); [x,~,~]=ti.features(zeros(120000,1),cfg);
verifyTrue(t,all(isfinite(x))); verifyEqual(t,x(41),0,'AbsTol',1e-12);
end

function testTrainOnlyScalingAndScores(t)
rng(7); cfg=ti.config(); y=repelem((1:11)',4); X=randn(44,4); X(:,1)=X(:,1)+y*3;
ids=compose("source_%d",(1:44)'); model=ti.train(X,y,ids,cfg);
verifyEqual(t,model.mu,mean(X,1),'AbsTol',1e-12);
[pred,p]=ti.predict(model,[X;1000*ones(1,4)]);
verifySize(t,p,[45 11]); verifyEqual(t,sum(p,2),ones(45,1),'AbsTol',1e-10);
verifyTrue(t,all(ismember(pred,1:11))); verifyEqual(t,model.mu,mean(X,1),'AbsTol',1e-12);
end

function testCacheOrder(t)
cfg=ti.config(); m=ti.manifest(cfg); segment_ids=flipud(m.segment_id);
X=flipud(repmat((1:988)',1,64)); metadata=struct('origin','synthetic unit test only');
path=[tempname,'.mat']; cleaner=onCleanup(@()delete(path)); %#ok<NASGU>
save(path,'X','segment_ids','metadata'); s=ti.cache(path,m,64);
verifyEqual(t,s.X(:,1),(1:988)');
end

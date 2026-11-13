function s = metrics(truth,guess,labels)
[ok,t]=ismember(truth,labels); assert(all(ok),'Unknown true label.');
[ok,p]=ismember(guess,labels); assert(all(ok),'Unknown predicted label.');
C=accumarray([t(:),p(:)],1,[numel(labels),numel(labels)]);
tp=diag(C); den=sum(C,2)+sum(C,1)'; f=zeros(size(tp));
f(den>0)=2*tp(den>0)./den(den>0);
s=struct('accuracy',mean(truth(:)==guess(:)),'macro_f1',mean(f),'confusion',C);
end

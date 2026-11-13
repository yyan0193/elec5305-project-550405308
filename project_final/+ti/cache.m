function s = cache(path,m,dimension)
s=load(path);
assert(isfield(s,'metadata'),'Cache must have provenance metadata.');
assert(isfield(s,'X')&&isfield(s,'segment_ids'),'Invalid feature MAT file.');
ids=string(s.segment_ids); ids=strip(ids(:));
assert(numel(unique(ids))==numel(ids),'Duplicate cache segment IDs.');
assert(size(s.X,2)==dimension&&size(s.X,1)==height(m),'Wrong feature dimensions.');
[ok,order]=ismember(m.segment_id,ids); assert(all(ok),'Cache does not cover manifest.');
s.X=double(s.X(order,:)); s.segment_ids=m.segment_id;
assert(all(isfinite(s.X(:))),'Cache contains NaN or Inf.');
end

function ab = mergestruct(a, b)
% Requires: a is a structure or a=[];   b is a structure
% Returns ab, the union of structures a, b
% For overlapping fields, the field of a takes precedence
% Usage ideas:  for merging optional parameter structures with 
%    default parameter structures
% loop through a and just assign b to it
% Example 
%» a.a = 1
% » b.b = 2
%» mergestructs(a,b)
%ans = 
%    b: 2
%    a: 1

% if any(ismember(fields(a), fields(b)) == 0), error('wrong input type'); end

ab = b;
if isempty(a), return; end
afields = fieldnames(a);

for i=1:length(afields)
   ab = setfield(ab, afields{i}, getfield(a, afields{i}));
end

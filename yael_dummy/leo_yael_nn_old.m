% Return the k nearest neighbors of a set of query vectors
%
% Usage: [ids,dis] = nn(v, q, k, distype)
%   v                the dataset to be searched (one vector per column)
%   q                the set of queries (one query per column)
%   k  (default:1)   the number of nearest neigbors we want
%   distype          distance type: 1=L1, 
%                                   2=L2         -> Warning: return the square L2 distance
%                                   3=chi-square -> Warning: return the square Chi-square
%                                   4=signed chis-square
%                                   16=cosine    -> Warning: return the *smallest* cosine 
%                                                   Use -query to obtain the largest
%                    available in Mex-version only
%
% Returned values
%   idx         the vector index of the nearest neighbors
%   dis         the corresponding *square* distances
%
% Both v and q contains vectors stored in columns, so transpose them if needed
function [idx, dis] = leo_yael_nn (X, Q, k, db_names,queryid,distype)


if ~exist('k'), k = 1; end
if ~exist('distype'), distype = 2; end 
assert (size (X, 1) == size (Q, 1));

switch distype
case {2,'L2'}
  
    
    %Top_ids_name = zeros(100,1);
    %for i=1:100
    %Top_ids_name(i,1) = db.dbImageFns(ids(i,1));
    %end
    
    
    base_addr = '/mnt/1E48BE700AFD16C7/datasets/output-files';
    query_name = db_names.qImageFns(queryid);
    Mat_fileq = char(strcat(base_addr,'/q/',query_name));
    Mat_fileq = strrep(Mat_fileq,'.jpg','.mat');

    query_qi = load(Mat_fileq);
    Q = query_qi.feats(:,1:1);
        sim = [];
    for jj = 1:length(db_names.dbImageFns)
        rr = fprintf('->%i->', jj);
        disp(rr);
        db_name = db_names.dbImageFns(jj);
        Mat_filedb = char(strcat(base_addr,'/db/',db_name));
        Mat_filedb = strrep(Mat_filedb,'.jpg','.mat');

        query_db = load(Mat_filedb);
        db_feats = query_db.feats;
        ds_all = [];

        for ii = 1:1
          % Compute half square norm
          X = db_feats(:,ii);
          X_nr = sum (X.^2) / 2;
          Q_nr = sum (Q.^2) / 2;

          sim_internal = bsxfun (@plus, Q_nr', bsxfun (@minus, X_nr, Q'*X)); 
          
          ds_all = [ds_all sim_internal];
          
        end
        y=sort(ds_all(:),'ascend');

        aa = sum(y(1:200))/200;
        sim = [sim,aa];
    end
    
      [dis, idx] = sort (sim, 2);
      dis = dis (:, 1:k);
      idx = idx (:, 1:k);
      

      dis = dis' * 2;
      idx = idx';
      

case {16,'COS'}
  sim = Q' * X;
                
  if k == 1
    [dis, idx] = min (sim, [], 2);
    dis = dis';
    idx = idx';
  else  
    [dis, idx] = sort (sim, 2);
    dis = dis (:, 1:k)';
    idx = idx (:, 1:k)';
  end
                 
otherwise
   error ('Unknown distance type');
end

                

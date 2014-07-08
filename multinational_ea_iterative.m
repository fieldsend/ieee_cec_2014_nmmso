function [P_real, P_eval, P_binary, nation_mask, evals] = ...
    multinational_ea_iterative(pop_size, max_evaluations, problem_func,...
    problem_func_params, param_num, mn, mx, select_type, ...
    evals, P_real, P_eval, P_binary, nation_mask, gov_size, p_mut, p_cross)


% Implementation of the "Multinational Evolutionary Algorithm"
% described by Rasmus K. Ursem (1999) in the paper by the same name in 
% Proceedings of the Congress on Evolutionary Computation, pages 1633-1640
%
% Implementation provided corresponds to that used in:
%"Running Up Those Hills: Multi-Modal Search with the Niching Migratory 
% Multi-Swarm Optimiser"
% by Jonathan E. Fieldsend
% published in Proceedings of the IEEE Congress on Evolutionary Computation, 
% pages 2593-2600, 2014
%
% Please reference both papers if you undertake work utilising this code.
% Implementation (c) by Jonathan Fieldsend, University of Exeter, 2014
%
% Assumes function maximisation
%
% REQUIRED ARGUMENTS
%
% pop_size = population size
% max_evaluations = maximum number of evaluations to be taken through the
%   problem function
% problem_func = string containing function to be optimised
% problem_func_params = meta-parameters needed by test function (distinct
%   from optimisation (design) parameters
% param_num = number of design parameters
% mn = minimum design parameter values (a vector with param_num elements)
% mx = maximum design parameter values (a vector with param_num elements)
% select_type = slection_type used. 1 for weighted, otherwise national 
%   selection is used
%
% OPTIONAL ARGUMENTS
%
% If evals argument is zero, or absent, then the optimiser starts from a 
% random population, otherwise will increment from the population specified 
% in the following optional arguments
%
% P_real = real values of world population, matrix of, pop_size by param_num
% P_eval = vector of function evalutions of world population, pop_size by 1
% P_binary = matrix of binary representation of world population, pop_size 
%   by (param_num*20)
% nation_mask = vector of nation meberships, pop_size by 1
% evals = how many problem function evaluations have been used so far
%
% finally the user may set the following optional arguments, if they are
% not provided, the default used in the original CEC paper will be used.
%
% gov_size = (max) size of goverment, default 8
% p_mut = probability of bit mutation, default 0.025
% p_cross = probability of crossover, default 0.9
%
% OUTPUTS
%
% P_real = matrix of real values of world population, pop_size by param_num
% P_eval = vector of function evalutions of world population, pop_size by 1
% P_binary = matrix of binary representation of world population, pop_size 
%   by (param_num*20)
% nation_mask = vector of nation meberships, pop_size by 1
% evals = how many problem function evaluations have been used so far


if exist('evals','var')==0
   evals = 0;
end

% set up algorithm
bits_per_value = 20;

%calculate corresponding gene vector and values
gene_conv=ones(bits_per_value,1);
for i=2:bits_per_value
    gene_conv(i)=2^(i-1);
end

% use default parameters used in original 1999 paper if not specified
if exist('gov_size','var')==0
   gov_size = 8;
end
if exist('p_mut','var')==0
   p_mut = 0.025;
end
if exist('p_cross','var')==0
   p_cross = 0.9;
end

migration_vector = [0.25, 0.5, 0.75];
merge_vector = [0.02 0.25, 0.5, 0.75, 0.98];

% start from scratch, or use optional arguments
if evals==0
    P_binary = initialisation(pop_size, param_num, bits_per_value); % world population
    P_real = zeros(pop_size,param_num);
    % mask indicating which nation each world population member is part of, all
    % in a single nation at the start
    nation_mask = ones(pop_size,1);
    P_eval = zeros(pop_size,1); % evaluation of world population members, pre allocate
    for i=1:pop_size
        P_real(i,:) = binary_to_real(P_binary(i,:),bits_per_value,gene_conv,mn,mx);
        P_eval(i) = feval(problem_func, P_real(i,:), problem_func_params);
    end
    evals = pop_size;
end

% until we hit the maximium number of evaluations
if (evals < max_evaluations)
    % elect and get policies for each nation
    
    [num_nat, national_policies, national_policies_eval, evals] = ...
       get_national_policies(gov_size, P_real, P_eval, nation_mask, ...
                                 problem_func, problem_func_params,param_num, evals);
    
    % migrate individuals
    
    % for each individual check if it should migrate based on its location and
    % the policy of its nation
    
    [nation_mask, national_policies, national_policies_eval, num_nat, evals] = ...
       migrate_individuals(pop_size, national_policies, national_policies_eval, ...
                          P_real, P_eval, nation_mask, migration_vector, ...
                          problem_func, problem_func_params, evals, num_nat);
    
     
    % merge entire nations
    % compare every nation to every other nation
    
    [nation_mask, evals] = ...
       merge_nations(national_policies, national_policies_eval, nation_mask, ...
                      merge_vector, problem_func, problem_func_params, evals, num_nat);
    
    
    % mate 
    
    [child_binary, child_real, child_eval, child_mask, evals ] = ...
            generate_offspring(nation_mask, P_binary, p_cross, p_mut, ...
                bits_per_value,gene_conv,mn,mx, pop_size, ...
                problem_func, problem_func_params, param_num, evals, num_nat);
    
    % combine parent and children into single population
    P_binary = [P_binary; child_binary];
    P_eval = [P_eval; child_eval];
    P_real = [P_real; child_real];
    nation_mask = [nation_mask; child_mask];
    
    % now select next population
    [parent_indices] = select(P_eval, nation_mask, pop_size, select_type, num_nat);
    
    % parent_indices  now contains indices of all selected parents.
    P_binary = P_binary(parent_indices,:);
    P_real = P_real(parent_indices,:);
    nation_mask = nation_mask(parent_indices); 
    P_eval = P_eval(parent_indices);
    
    
    % check that no nation has emptied -- possible as mean is taken, so if
    % e.g. nation was of two, both members may find valley between them and
    % mean, so both may have exited (not mentioned in original paper)
    
    num_nat = max(nation_mask);
    for k=1:num_nat;
        if (sum(nation_mask==k)==0)
            nation_mask(nation_mask>k) = nation_mask(nation_mask>k) -1;
        end
    end
    
    fprintf('Evals %d, nations %d, best solution %f\n',evals, max(nation_mask), max(P_eval));
end

%------------
function [num_nat, national_policies, national_policies_eval, evals] = ...
    get_national_policies(gov_size, P_real, P_eval, nation_mask, ...
    problem_func, problem_func_params, param_num, evals)

% fixes the policy for each nation in this time step

num_nat = max(nation_mask);

national_policies = zeros(num_nat,param_num);
national_policies_eval = zeros(num_nat,1);
for k=1:num_nat
    % get the national policy for each of the nations
    national_policies(k,:) = election_policy(gov_size, P_real(nation_mask==k,:), P_eval(nation_mask==k));
    % need to evaluate each national policy too, as used in migration
    % check
    national_policies_eval(k) = feval(problem_func, national_policies(k,:), problem_func_params);
end
evals = evals + num_nat;


%------------
function [nation_mask, national_policies, national_policies_eval, num_nat, evals] = ...
    migrate_individuals(pop_size, national_policies, national_policies_eval, ...
    P_real, P_eval, nation_mask, migration_vector, ...
    problem_func, problem_func_params, evals, num_nat)

% function migrates the individuals away from their current nations and
% to other nations (or their own start up nation) if there are any valleys
% detected between them and their government policy location

for i=1:pop_size
    [valley, calls] = hill_valley(P_real(i,:), ...
        national_policies(nation_mask(i),:),P_eval(i), ...
        national_policies_eval(nation_mask(i)), migration_vector, ...
        problem_func, problem_func_params);
    evals = evals + calls;
    
    if valley > 0 % shouldn't be on current peak
        [move, calls] = migration_check(P_real, P_eval, i,...
            national_policies, national_policies_eval, nation_mask, ...
            migration_vector, problem_func, problem_func_params);
        evals = evals + calls;
        if move > 0 % move nation membership to another nation
            % paper does not state explictly how policy change
            % should be handled when nations merge, however, as
            % it is indexed by t, it is inferred there is only
            % one policy per nation per time step, so it is not
            % recalculated each time an individual
            % enters/leaves a nation during a time step
            
            nation_mask(i) = move;
        else % make new nation, policy set as location
            nation_mask(i) = max(nation_mask) + 1;
            % In a new nation of a single member, the policy is that member
            national_policies(end+1,:) = P_real(i,:);
            national_policies_eval(end+1) = P_eval(i);
            num_nat = num_nat + 1;
        end
    end
end


%------------
function [nation_mask, evals] = merge_nations(national_policies, ...
              national_policies_eval,  nation_mask, merge_vector, ...
              problem_func, problem_func_params, evals,num_nat)

% merge nations together, if appropriate          
          
for k=num_nat:-1:1
    % compare each nation to each other (untill a merge is detected or nations run out)
    if sum(nation_mask==k)>0
        %[~,sorted_nations] = sort(national_policies,'descend');
        for j=max(nation_mask):-1:1
            % if compared nation is not current nation, and current nation has members
            if (j~=k)
                [move_val, calls] = hill_valley(national_policies(k,:), national_policies(j,:),...
                    national_policies_eval(k), national_policies_eval(j), merge_vector, problem_func, problem_func_params);
                evals = evals + calls;
                if (move_val == 0)
                    % paper does not state explictly how policy change
                    % should be handled when nations merge, however, as
                    % it is indexed by t, it is inferred there is only
                    % one policy per nation per time step, so it is not
                    % recalculated each time an individual
                    % enters/leaves a nation during a time step
                    
                    nation_mask(nation_mask == k) = j; % move all from k nation to j
                    
                    % decrease nation number of all nations above removed
                    % nation by one, to shunt them down and remove nation k
                    nation_mask(nation_mask>k) = nation_mask(nation_mask>k)-1;
                    % expunge defunct policies
                    national_policies(k,:)=[];
                    national_policies_eval(k)=[];
                    
                    % already found one to merge to so no need to compare
                    % to any remaining
                    break;
                end
            end
        end
    else
        % empty nation - so remove
        nation_mask(nation_mask>k) = nation_mask(nation_mask>k)-1;
        % expunge defunct policies
        national_policies(k,:)=[];
        national_policies_eval(k)=[];
    end
end

%------------
function [child_binary, child_real, child_eval, child_mask, evals ] = ...
            generate_offspring(nation_mask, P_binary, p_cross, p_mut, ...
                bits_per_value,gene_conv,mn,mx, pop_size, ...
                problem_func, problem_func_params, param_num, evals, num_nat)
          
% generate children          
          
child_binary = zeros(pop_size,param_num*bits_per_value);
child_mask = zeros(pop_size,1);
i=1;
% mating rule is only members of same nation may mate
for k = 1:num_nat
    I = find(nation_mask == k);
    if length(I) > 1 % if crossover feasible, at least two national parents
        II = randperm(length(I)); % randomly select parents in nation
        if rem(length(II),2)==1
            % odd number of parents -- so last will just be duplicated to mutate
            child_binary(I(II(end)),:) = P_binary(I(II(end)),:);
        end
        % now crossover all other parents
        for j=1:2:length(II)-1
            [child_binary(I(II(j)),:), child_binary(I(II(j+1)),:)] = ...
                crossover(P_binary(I(II(j)),:), P_binary(I(II(j+1)),:), p_cross);
        end
        child_mask(i:i+(length(I)-1)) = k;
        i = i + length(I);
    elseif length(I) == 1 % just one nation member, so simply copy -- it will be mutated
        child_binary(I(1),:) = P_binary(I(1),:);
        child_mask(i) = k;
        i = i + 1;
    end
end
child_binary = mutate(child_binary,p_mut);

child_real = zeros(size(child_binary,1),size(child_binary,2)/bits_per_value);
child_eval = zeros(size(child_binary,1),1);

% evaluate children;
for i=1:pop_size
    child_real(i,:) = binary_to_real(child_binary(i,:),bits_per_value,gene_conv,mn,mx);
    child_eval(i) = feval(problem_func, child_real(i,:), problem_func_params);
end
evals = evals + pop_size;


%------------
function [parent_indices] = select(P_eval, nation_mask, pop_size, select_type, num_nat)

% select the pop_size parents to go into the next generation from the 
% pop_size*2 current population of previous parents and current children

parent_indices = zeros(pop_size,1);

if select_type == 1 % weighted selection
    % calculate size of each nation 
    nation_size = zeros(num_nat,1);
    for k=1:num_nat
        nation_size(k) = sum(nation_mask==k);
    end
    j=1;
    for i=1:pop_size*2
        r = randperm(pop_size);
        % choose parent based on weighted fitness
        if P_eval(r(1))/nation_size(nation_mask(r(1))) > P_eval(r(2))/nation_size(nation_mask(r(2)))
            parent_indices(j) = r(1);
        else
            parent_indices(j) = r(2);
        end
        j = j+1;
    end
    % recalculate size of each nation
    nation_size = zeros(num_nat,1);
    for k=1:num_nat
        nation_size(k) = sum(nation_mask==k);
    end
else % national selection
    i=1;
    for k=1:num_nat
        I = find(nation_mask == k);
        r = randperm(length(I));
        for j = 1:2:length(I)-1 % iterate over each nation and subsample fittest to make parents
            if P_eval(I(r(j))) > P_eval(I(r(j+1)))
                parent_indices(i) = I(r(j));
            else
                parent_indices(i) = I(r(j+1));
            end
            i=i+1;
        end
    end
end

%------------
function [move,calls] = migration_check(P_real,P_eval,index,national_policies,national_policies_eval, nation_mask, migration_vector, problem_func, problem_func_params)

move=0;
calls=0;
for k=1:max(nation_mask)
    if (nation_mask(index)~=k) % don't recompare to its current nation
        [move_val, intermediate_calls] = hill_valley(P_real(index,:), national_policies(k,:),...
            P_eval(index), national_policies_eval(k), migration_vector, problem_func, problem_func_params);
        calls = calls + intermediate_calls;
        if (move_val==0)
            % found an acceptable nation to move to
            move = k;
            % as found a nation to merge into, no need to compare to
            % any remaining nations
            return;
        end
    end
end
%------------
function pl = election_policy(gov_size, population, population_evals)

% determine government based upon fitness

if size(population,1)<=gov_size
    government = population;
else
    [~,I] = sort(population_evals,'descend');
    government = population(I(1:gov_size),:);
end
if size(population,1)==1
    pl = government;
else
    pl = mean(government,1);
end
    
%------------
function [c1,c2] = crossover(p1,p2,p_cross)

% crossover type not mentioned in paper, single point used
c1=p1;
c2=p2;
if rand()<p_cross
   slice_point = randperm(length(p1)-1);
   c1(slice_point+1:end) = p2(slice_point+1:end);
   c2(slice_point+1:end) = p1(slice_point+1:end);
end


%------------
function [c] = mutate(p,p_mut)

r = rand(size(p)) < p_mut; % find those to flip
c = p + r; % flip 0 to 1, and 1 to 2
c(c > 1) = 0; % flip the 2 to 0 as required

%------------
function P_binary = initialisation(pop_size, param_num, bits_per_value)

P_binary = rand(pop_size,param_num*bits_per_value);
P_binary = P_binary > 0.5;


%------------
function [v, calls, pts] = hill_valley(i1, i2, fit1, fit2, samples, problem_func, problem_func_params)

% returns 0 if all sampled points on line between i1 and i2 arguments
% have are equal to or higher than the minimum response of i1 and i2


min_fit = min(fit1,fit2);
v = 0;
calls=0; % keep track of the number of problem function evaluations
pts  = zeros(size(samples));
for j = 1: length(samples);
    % generate point on line between points
    interior_j = i1 +(i2-i1)*samples(j);
    pts(j) = feval(problem_func, interior_j, problem_func_params);
    calls = calls + 1;
    if min_fit > pts(j)
        v = min_fit - pts(j);
        return;
    else
        pts(j) = 0;
    end
end

%----------------------------------------------
function x = binary_to_real(y,gene_length,g,mn,mx)

% function x = binary_to_real(y,gene_length)
% converts the binary vector 'y' of length 'gene_length' into a real value
% [0,1]


num=length(y);
if rem(length(y),gene_length)~=0
    error(['Binary decision variable length is not divisible by gene size' ...
        ' provided']);
end

p=num/gene_length;
x=zeros(1,p);

sum_g=sum(g);
j=1;
%y(j:j+gene_length-1)
%g
for i=1:p
    x(i)=(y(j:j+gene_length-1)*g)/sum_g;
    j=j+gene_length;
end

% now scale from 0 to 1 on each dimension, to between mn and mx
rng = mx-mn;
x = mn+ x.*rng;


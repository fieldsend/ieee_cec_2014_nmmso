function [P_real_before, P_eval_before, P_binary_before, ...
    nation_mask_before, evals_before, P_real_after, P_eval_after, ... 
    P_binary_after, nation_mask_after, evals_after] = ...
    multinational_ea(pop_size, max_evaluations, problem_func,...
    problem_func_params, param_num, mn, mx, select_type, ...
    gov_size, p_mut, p_cross)

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
% The user may set the following optional arguments, if they are
% not provided, the default used in the original CEC paper will be used.
%
% gov_size = (max) size of goverment, default 8
% p_mut = probability of bit mutation, default 0.025
% p_cross = probability of crossover, default 0.9
%
% OUTPUTS
%
% Due to the algorithm design (dynamic populations) the final generation
% may exceed the alloted maximum number of evaluations. As such the final
% population state and the penultimate population state are returned (with 
% the corresponding evaluation number tracked)
%
% P_real = matrix of real values of world population, pop_size by param_num
% P_eval = vector of function evalutions of world population, pop_size by 1
% P_binary = matrix of binary representation of world population, pop_size 
%   by (param_num*20)
% nation_mask = vector of nation meberships, pop_size by 1
% evals = how many problem function evaluations have been used so far
% P_real = matrix of real values of world population, pop_size by param_num
%
% P_eval = vector of function evalutions of world population, pop_size by 1
% P_binary = matrix of binary representation of world population, pop_size 
%   by (param_num*20)
% nation_mask = vector of nation meberships, pop_size by 1
% evals = how many problem function evaluations have been used so far

if (max_evaluations<=0)
   error('max_evaluations argument must be positive'); 
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

% at start no evaluations used, and multinational EA state is empty
P_real_after=[];
P_eval_after=[];
P_binary_after=[];
nation_mask_after=[];
evals_after=0;


while (evals_after < max_evaluations)
    P_real_before = P_real_after;
    P_eval_before = P_eval_after;
    P_binary_before = P_binary_after;
    nation_mask_before=nation_mask_after;
    evals_before = evals_after; 
 
    [P_real_after, P_eval_after, P_binary_after, nation_mask_after, evals_after] = ...
        multinational_ea_iterative(pop_size, max_evaluations, problem_func, ...
    problem_func_params, param_num, mn, mx, select_type, evals_after, ...
    P_real_after, P_eval_after, P_binary_after, nation_mask_after, gov_size, p_mut,p_cross);
end



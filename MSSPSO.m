function [evals_before, swarm_before, X_before, Y_before, evals_after, ...
    swarm_after, X_after, Y_after] = ...
    MSSPSO(swarm_size, swarm_num, max_evaluations, problem_func,...
    problem_func_params, param_num, mn, mx)


% Implementation of the algorithm described in:
% "Multi-Sub-Swarm Particle Swarm Optimization Algorithm for Multimodal 
% Function Optimization" 
% by Jun Zhang et al.
% published in Proceedings of the IEEE Congress on Evolutionary Computation, 
% pages 3215-3220, 2007
%
% Implementation provided corresponds to that used in:
%"Running Up Those Hills: Multi-Modal Search with the Niching Migratory 
% Multi-Swarm Optimiser"
% by Jonathan E. Fieldsend
% published in Proceedings of the IEEE Congress on Evolutionary Computation, 
% pages 2593-2600, 2014
%
% Please reference both papers if you undertake work utilising this code.
%
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
%
% OUTPUTS
%
% Due to the algorithms design, repeated generations will often not lead to
% exactly 'evals' function evaluations. As such it returns the state of the 
% algorithm in the generation before the evals limit is breached, and 
% the state after it is breached  
%
% evals_before = number of function evaluations used but generation
% directly prior to limit breach
% swarm_before = structure containing state of swarm(s) at 'evals_before' function
% evaluations
% X_before = matrix containing location of all the swarms pbests (first half) 
% and current particle locations (second half) at evals_before
% Y_before = function response at X locations at evals_before
%
% evals_after = number of function evaluations used but generation
% directly prior to limit breach
% swarm_after = structure containing state of swarm(s) at 'evals_after' function
% evaluations
% X_after = matrix containing location of all the swarms pbests (first half) 
% and current particle locations (second half) at evals_after
% Y_after = function response at X locations at evals_after

if (max_evaluations<=0)
   error('max_evaluations argument must be positive'); 
end

% at start no evaluations used, and swarm structure is empty
evals_after = 0;
swarm_after = [];
X_after=[];
Y_after=[];

while (evals_after < max_evaluations)
    evals_before = evals_after;
    swarm_before = swarm_after;
    X_before = X_after;
    Y_before = Y_after;    
    [evals_after, swarm_after, X_after, Y_after] = ...
            MSSPSO_iterative(swarm_size, swarm_num, max_evaluations, problem_func,...
    problem_func_params, param_num, mn, mx, evals_after, swarm_after);
end

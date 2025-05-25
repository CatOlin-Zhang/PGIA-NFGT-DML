clc;
clear;
% Example usage
V = [
    0     5     8     0     0;
    0     0     0     3     4;
    0     2     0    10     0;
    0     0     0     0     8;
    0     0     0     0     0
];

C = [
     0     8     7     0     0;
     0     0     0     2     9;
     0     5     0     9     0;
     0     0     0     0     4;
     0     0     0     0     0
 ];

[flow, minCost] = networkSimplex(V, C);
disp('Flow:');
disp(full(flow));
disp(['Minimum Cost: ', num2str(minCost)]);
function [flow, minCost] = networkSimplex(V, C)
    % V: Capacity matrix
    % C: Cost matrix
    
    n = size(V, 1);
    m = nnz(triu(V)); % Number of edges in the graph
    flow = sparse(n, n); % Initialize flow matrix as a sparse matrix
    basis = zeros(m, 1); % Basis vector to store indices of basic variables
    cost = C(find(triu(C))); % Flatten cost matrix into a column vector for non-zero entries
    capacity = V(find(triu(V))); % Flatten capacity matrix into a column vector for non-zero entries
    
    % Create an index map for non-zero entries in V
    [i, j] = find(triu(V));
    
    A = spalloc(2*n-2, m, 3*m); % Incidence matrix
    
    % Fill the incidence matrix A
    rowIdx = 1;
    edgeIdx = 1; % Index for edges
    for k = 1:m
        u = i(k);
        v = j(k);
        
        if u ~= v
            A(rowIdx, edgeIdx) = -1; % Outgoing edge from node u
            A(rowIdx+1, edgeIdx) = 1; % Incoming edge to node v
            
            if u == 1 || v == 1
                basis(edgeIdx) = edgeIdx; % Include edge in the initial basis
                flow(u, v) = capacity(edgeIdx); % Set flow along the edge to its capacity
            end
            
            rowIdx = rowIdx + 2;
            edgeIdx = edgeIdx + 1;
        end
    end
    
    % Remove rows corresponding to the root node (node 1)
    A = A(3:end, :);
    
    % Initialize dual prices pi and reduced costs rc
    pi = zeros(n, 1);
    pi(1) = inf; % Dual price for the root node is infinity
    rc = cost - pi(i) + pi(j);
    
    while any(rc(basis(basis > 0)) < 0)
        enteringIdx = find(rc(basis(basis > 0)) < 0, 1); % Choose an entering variable
        
        % Perform pricing step to determine the leaving variable
        [~, path] = shortestPath(A(:, basis(basis > 0)), enteringIdx);
        leavingIdx = path(end);
        
        % Determine the direction of circulation and the minimum capacity
        delta = min(flow(i(leavingIdx)), ...
                    capacity(enteringIdx) - flow(i(enteringIdx)));
        
        if delta <= 0
            error('Infeasible solution detected.');
        end
        
        % Update flows along the cycle
        flow(i(leavingIdx), j(leavingIdx)) = flow(i(leavingIdx), j(leavingIdx)) - delta;
        flow(i(enteringIdx), j(enteringIdx)) = flow(i(enteringIdx), j(enteringIdx)) + delta;
        
        % Update the basis
        temp = basis(leavingIdx);
        basis(leavingIdx) = basis(enteringIdx);
        basis(enteringIdx) = temp;
        
        % Update dual prices
        pi(i(leavingIdx)) = pi(i(enteringIdx));
        
        % Recalculate reduced costs
        rc = cost - pi(i) + pi(j);
    end
    
    % Calculate the minimum cost
    minCost = sum(cost .* full(flow(:)));
end

% Helper function to find the shortest path using Dijkstra's algorithm
function [dist, path] = shortestPath(A, s)
    n = size(A, 1);
    dist = inf(n, 1);
    dist(s) = 0;
    visited = false(n, 1);
    prev = zeros(n, 1);
    
    pq = PriorityQueue();
    pq.push(s, 0);
    
    while ~pq.isEmpty()
        [~, u] = pq.pop();
        visited(u) = true;
        
        for v = find(A(:, u))
            alt = dist(u) + A(v, u);
            
            if ~visited(v) && alt < dist(v)
                dist(v) = alt;
                prev(v) = u;
                pq.push(v, alt);
            end
        end
    end
    
    % Reconstruct the path
    path = [];
    u = s;
    while u ~= 0
        path = [u, path];
        u = prev(u);
    end
end





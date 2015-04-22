function x = inv_triu(U)
x = solve_triu(U,eye(size(U)));

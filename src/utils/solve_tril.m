function x = solve_tril(T,b)
opt.LT = true;
x = linsolve(T, b, opt);
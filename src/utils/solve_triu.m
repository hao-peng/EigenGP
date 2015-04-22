function x = solve_triu(T,b)
opt.UT = true;
x = linsolve(T, b, opt);
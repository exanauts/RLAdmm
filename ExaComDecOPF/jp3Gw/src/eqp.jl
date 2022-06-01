using SparseArrays
using LinearAlgebra
using Printf

"""
solve_eqp(zsol, G, At, rhs)

Solve the following form of equality constrained symmetric QP using
either full-range or nullspace method.

    Min 0.5*x*G*x - c*x s.t. At*x = b

Input:
  G   = the lower-triangular part of G
  At  = the constrained matrix
  rhs = [ c ; b]

Output:
  zsol = [ xsol ; ysol ] where xsol and ysol are the primal and dual
         solutions, respectively.
"""
function solve_eqp(zsol::Vector{Float64},
                   G::SparseMatrixCSC{Float64},
                   At::SparseMatrixCSC{Float64},
                   rhs::Vector{Float64};
                   method=:FullRange, verbose=0)
    m,n = size(At)
    @assert n == size(G,2)

    A = sparse(transpose(At))
    Gf = G + transpose(G) - Diagonal(G)
    K = [ Gf A; At spzeros(m,m) ]

    # Calculate a solution using the full-range method.
    if method == :FullRange
        # We'll replace LU with LDLt to detect the inertia of K in the future.
        F = lu(K)
        zf = F.U \ (F.L \ (F.Rs .* rhs)[F.p])
        zf = zf[invperm(F.q)] # A solution to K*z = rhs.
        if verbose > 0
            err_f = norm(K * zf .- rhs)
            @printf("error of full-range = %.5e\n", err_f)
        end

        zsol .= zf
    elseif method == :Nullspace
        # Calculate a solution using the nullspace method.
        # All computations will be done in the permuted and scaled space.

        # (Rs .* A)[p,q] = L * U.
        # -> (tran(A) .* tran(Rs))[q,p] = tran(U) * tran(L) or
        # tran(A) = ( (tran(U) * tran(L))[inv(q),inv(p)] ) ./ tran(Rs).
        F = lu(A)

        # 0. Calculate a nullspace of tran(A)_hat = (tran(A) .* tran(Rs))[q,p].
        L1 = F.L[1:m,:]
        L2 = F.L[m+1:end,:]
        Z = [ -transpose(L1) \ transpose(L2) ; Diagonal(ones(n-m)) ]
        #@assert norm(At * (F.Rs .* Z[invperm(F.p),:])) <= 1e-10

        # 1. Calculate the reduced Hessian.
        H = (F.Rs .* Gf .* transpose(F.Rs))[F.p,F.p]
        rH = transpose(Z) * H * Z
        rH = (rH + transpose(rH)) / 2 # Force symmetricity.
        rm, rn = size(rH)
        @assert rn == (n-m)

        # 2. Calculate a particular solution satisfying tran(A)_hat*s_hat=b[q].
        # The solution in the original space is Rs .* s[inv_p].
        s = [ transpose(L1) \ (transpose(F.U) \ ((rhs[n+1:end])[F.q])) ; zeros(n-m) ]

        # 3. Calculate t = c - H * s
        t = (F.Rs .* rhs[1:n])[F.p] .- (H * s)

        # 4. Calculate u2 = tran(Z)*t = t2 - L2*L1^{-1}*t1.
        u2 = t[m+1:end] - L2 * (L1 \ t[1:m])

        # 5. Solve rH * v = u2.
        cF = cholesky(rH)
        v = cF \ u2
        #@assert norm(rH * v - u2) <= 1e-10

        # 6. Calaulte w = Z * v = [-tran(L1)^{-1} * tran(L2) * v ; v ].
        w = Z * v

        # 7. Calculate x = s + w.
        x = s .+ w

        # 8. Calculate g = c - H * x.
        g = (F.Rs .* rhs[1:n])[F.p] - H * x

        # 9. Calculate y = U^{-1} * L1^{-1} * g.
        y = F.U \ (L1 \ g[1:m])

        # Restore the permutation and unscale.
        zn = [ F.Rs .* x[invperm(F.p)]; y[invperm(F.q)] ]
        if verbose > 0
            err_n = norm(K * zn .- rhs)
            @printf("error of nullspace  = %.5e\n", err_n)
        end

        zsol .= zn
    else
        error("ERROR: unknown method for EQP solve: ", method)
    end

    return
end

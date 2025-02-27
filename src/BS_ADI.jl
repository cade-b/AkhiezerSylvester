using LinearAlgebra, SparseArrays, EllipticFunctions, BlockDiagonals
#all functions ported from MATLAB to Julia by Cade Ballew, Sept. 2024 (cadeballew@gmail.com).

function bartelsStewart(A, B, C, D, E, split=[false ; false])
#=BARTELSSTEWART  Solve generalized Sylvester matrix equation. 
   BARTELSSTEWART(A, B, C, D, E) solves the generalized Sylvester equation

         AXB^T + CXD^T = E

   using the Bartels--Stewart algorithm [1,2].

   BARTELSSTEWART(A, [], [], D, E) assumes B = I and C = I. This allows more
   efficient computation than passing identity matrices. Similarly,
   BARTELSSTEWART(A, B, [], [], E) assumes C = B and D = A.
 
   BARTELSSTEWART(A, B, C, D, E, SPLIT) also takes information SPLIT which
   states whether or not the problem may be decoupled into even and odd modes
   for efficiency.

   References:
    [1] R. H. Bartels & G. W. Stewart, Solution of the matrix equation 
    AX +XB = C, Comm. ACM, 15, 820–826, 1972.
 
    [2] J. D. Gardiner, A. J. Laub, J. J. Amato, & C. B. Moler, Solution of the
    Sylvester matrix equation AXB^T + CXD^T = E, ACM Transactions on
    Mathematical Software (TOMS), 18(2), 223-231, 1992.

 Copyright 2014

 Written by Alex Townsend, Sept 2012. (alex.townsend1987@gmail.com)
 Modified by Nick Hale, Nov 2014. (nick.p.hale@gmail.com)=#

    tol = 10 * eps()
    
    # If the RHS is zero then the solution is the zero solution (assuming uniqueness).
    if norm(E) < tol
        return zeros(size(E))
    end

    # Special case: Standard Lyapunov and Sylvester equations
    if isempty(B) && isempty(C)
        if isempty(D)
            return lyap(A, -E)
        end
        return sylvester(A, transpose(D), -E)
    end

    # Matrices must be full for SCHUR():
    A = Matrix(A); B = Matrix(B); C = Matrix(C); D = Matrix(D)

    m, n = size(E)
    Y = zeros(m, n)

    # Look for problems of the form AXB^T + BXA^T = E
    AEQ = isempty(C) && isempty(D)
    if AEQ
        C = B
    end

    if isempty(C)
        P, Z1 = schur(A)
        Q1 = Z1'
        S = I(m)
    elseif split[2]
        # Implement split QZ factorization (calls qzSplit)
        P, S, Q1, Z1 = qzSplit(A, C)
    else
        P, S, Q1, Z1 = schur(A, C)
        Q1 = Q1' # Transpose to match MATLAB convention
    end

    if AEQ
        T = P
        R = S
        Q2 = Q1
        Z2 = Z1
    elseif isempty(B)
        T, Z2 = schur(D)
        Q2 = Z2'
        R = eye(n)
    elseif split[1]
        # If the PDE is even/odd in the y-direction then we can split (further) into double as many subproblems.
        T, R, Q2, Z2 = qzSplit(D, B)
    else
        T, R, Q2, Z2 = schur(D, B)
        Q2 = Q2'; # Transpose to match MATLAB convention
    end

    #= Now use the generalised Bartels--Stewart solver found in Gardiner et al.
       (1992). The Sylvester matrix equation now contains quasi upper-triangular
       matrices and we can do a backwards substitution. =#

    # Transform the righthand side.
    F = Q1*E*transpose(Q2)

    #Initialise S*Y and P*Y factors:
    PY = zeros(m, n)
    SY = zeros(m, n)

    # Do a backwards substitution type algorithm to construct the solution.
    k = n

    # Construct columns n,n-1,...,3,2,1 of the transformed solution.
    while  k > 0 
        
        #= There are two cases, either the subdiagonal contains a zero, i.e.,
           T(k,k-1) = 0 and then it is simply a backwards substitution, or T(k,k-1)
           != 0 and we solve a 2mx2m system. =#
        
        if  k == 1 || T[k,k-1] == 0 
            # Simple case (Usually end up here).
            
            jj = (k+1):n
            if k == n
                rhs = F[:,k]
            else
                rhs = F[:,k] - PY[:,jj]*R[k,jj] - SY[:,jj]*T[k,jj]
            end
            
            # Find the kth column of the transformed solution.
            tmp = (P + (T[k,k]/R[k,k])*S) # <- Divide both sides by R_kk for speed.
            rhs = rhs/R[k,k]
            Y[:,k] = tmp \ rhs
            
            # Store S*Y and P*Y factors:
            PY[:,k] = P*Y[:,k]
            SY[:,k] = S*Y[:,k]
            
            # Go to next column:
            k = k - 1;
            
        else

            #= This is a straight copy from the Gardiner et al. paper, and just
               solves for two columns at once. (Works because of quasi-triangular
               matrices.) =#
            
            # Operator reduction.
            jj = (k+1):n
            rhs1 = F[:,k-1] - PY[:,jj]*R[k-1,jj] - SY[:,jj]*T[k-1,jj]
            rhs2 = F[:,k]   - PY[:,jj]*R[k,jj]   - SY[:,jj]*T[k,jj]

            # 2 by 2 system.
            SM = [R[k-1,k-1]*P + T[k-1,k-1]*S R[k-1,k]*P + T[k-1,k]*S; T[k,k-1]*S R[k,k]*P + T[k,k]*S]

            # Solve (permute the columns and rows):
            idx = vec([(1:m)' ; (m+1:2m)'])
            rhs = [rhs1 ; rhs2]
            UM = SM[idx,idx] \ rhs[idx]
            UM[idx] = UM
            
            # Store S*Y and P*Y factors:
            Y[:,k-1:k] = reshape(UM, m, 2);
            PY[:,k-1:k] = P*Y[:,k-1:k];
            SY[:,k-1:k] = S*Y[:,k-1:k];

            # We solved for two columns so go two columns farther.
            k = k - 2;
            
        end
        
    end

    # We have now computed the transformed solution so we just transform it back.
    return Z1 * Y * Z2'
end

function qzSplit(A, C)
    # A faster QZ (Schur) factorization for problems that decouple
    # QZSPLIT() is equivalent to standard QZ, except we take account of symmetry to reduce the computational requirements.

    # Matrix size (square):
    n = size(A, 1)

    # Do the QZ by splitting the problem into two subproblems. 

    # Odd part:
    odd = 1:2:n
    A1 = A[odd, odd]
    C1 = C[odd, odd]
    P1, S1, Q1, Z1 = schur(A1, C1)
    Q1 = Q1' # Transpose to match MATLAB convention

    # Even part:
    even = 2:2:n
    A2 = A[even, even]
    C2 = C[even, even]
    P2, S2, Q2, Z2 = schur(A2, C2)
    Q2 = Q2' # Transpose to match MATLAB convention

    # Recombine the subproblems
    return qzRecombine(P1, P2, S1, S2, Q1, Q2, Z1, Z2)
end

function qzRecombine(P1, P2, S1, S2, Q1, Q2, Z1, Z2)
    hf1 = size(P1, 1)
    n = 2hf1 - 1
    top = 1:hf1
    bot = (hf1+1):n
    odd = 1:2:n
    even = 2:2:n

    P = BlockDiagonal([P1, P2])
    S = BlockDiagonal([S1, S2])

    Q = zeros(n, n)
    Q[top, odd] = Q1
    Q[bot, even] = Q2

    Z = zeros(n, n)
    Z[odd, top] = Z1
    Z[even, bot] = Z2

    return P, S, Q, Z
end

function adi(A, B, F, p=nothing, q=nothing; tol=eps())
#=adi(A, B, F, p=P, q=Q)
 approximately solves AX - XB = F using ADI with ADI shift parameters 
 provided by vectors P, Q.

 adi(A, B, F) If A and B have real eigenvalues that are contained in 
 disjoint intervals, the optimal shift parameters are computed automatically
 and the problem is solved to a relative accuracy of approximately machine 
 precision. 

 adi(A, B, F; tol=Tol) is as above, except the relative accuracy of the 
 of the solution is specified by Tol. 

 See getshifts_adi and getshifts_smith for help computing shift parameters. 

 References: 

 [1] Lu, An, and Eugene L. Wachspress. 
  "Solution of Lyapunov equations by alternating direction implicit iteration." 
  Comp. & Math. with Appl., 21.9 (1991): pp. 43-58.
 

 written by Heather Wilber (heatherw3521@gmail.com)
 Jan, 2018. =#
    m, n = size(F)
    #In = spdiagm(0 => ones(n))  # Sparse identity matrix
    #Im = spdiagm(0 => ones(m))  # Sparse identity matrix
    
    compute_shifts = (p === nothing || q === nothing)
    # user wants shift parameters computed:
    if compute_shifts
        # find intervals where eigenvalues live: 
        a = eigvals(A)[1]
        b = eigvals(A)[end]
        c = eigvals(B)[1]
        d = eigvals(B)[end]

        # Check if eigenvalues have a complex part
        if any(abs.(imag.([a, b, c, d])) .> 1e-10)
            error("ADI:adi:cannot automatically compute shift parameters unless the eigenvalues of A and B in AX - XB = F are contained in real, disjoint intervals.")
        end

        # Check if intervals overlap
        I1 = [min(a, b), max(a, b)]
        I2 = [min(c, d), max(c, d)]
        II = [a,b,c,d]
        if (I1[1] < I2[2] && I1[2] > I2[1]) || (I2[1] < I1[2] && I2[2] > I1[1])
            error("ADI:adi: cannot automatically compute shift parameters unless the eigenvalues of A and B in AX - XB = F are contained in real, disjoint intervals.")
        end
        p, q = getshifts_adi(II, tol)#getshifts_adi(I1, I2, tol)
    else
        # Ensure that p and q are provided and valid
        if length(p) != length(q)
            error("ADI:adi: length of p and q must be the same.")
        end
    end

    # do ADI
    X = zeros(m, n)
    for i = 1:length(p)
        X = (A - q[i] * I) \ (X * (B - q[i] * I) + F)
        X = ((A - p[i] * I) * X - F) / (B - p[i] * I)
    end

    return X
end

function getshifts_adi(II, N_or_tol)
    #=%
    % computes optimal ADI shift parameters for solving AX - XB = F
    % where I = [a b c d], and spectrum(A) \cup [a, b] and spectrum(B) \cup [c, d]. 
    % 
    % getshifts_adi(I, N::Integer) computes N ADI shift parameters. 
    % 
    % getshifts_adi(I, tol::Float) computes as many shift parameters 
    % as needed so that when they are used with adi or fadi, 
    % ||X_approx||_2 < ||X||*tol
    %
    % See also: getshifts_fiadi, getshifts_smith
    %
    % References: 
    %  [1] Lu, An, and Eugene L. Wachspress. 
    %  "Solution of Lyapunov equations by alternating direction implicit iteration." 
    %  Comp. & Math. with Appl., 21.9 (1991): pp. 43-58.
    %
    %  [2] B. Beckermann and A. Townsend, 
    %  "On the singular values of matrices with displacement structure."
    %  SIMAX, 38 (2017): pp. 1227-1248. 
    
    % written by Heather Wilber (heatherw3521@gmail.com)
    % Jan, 2018
    %%=#

    a, b, c, d = II

    # Check if intervals are overlapping:
    I1 = (min(a,b), max(a,b))
    I2 = (min(c,d), max(c,d))
    if (I1[1] < I2[2] && I1[2] > I2[1]) || (I2[1] < I1[2] && I2[2] > I1[1])
        error("ADI:getshifts_adi: The intervals containing the eigenvalues of A and B must be disjoint.")
    end

    # Compute the Möbius transform
    ~, Tinv, gam, cr = mobiusT(II)

    # Determine if input is tolerance or number of shifts
    if isa(N_or_tol, AbstractFloat)  # Tolerance is provided
        tol = N_or_tol
        N = ceil(Int, (1/pi^2) * log(4/tol) * log(16 * cr))
    elseif isinteger(N_or_tol)# Number of shifts is provided
        N = N_or_tol
    else
        error("ADI:getshifts_adi: N must be an integer or tol must be a float.")
    end

    # we estimate elliptic integrals when 1/gam is small
    if gam > 1e7
        K = (2*log(2) + log(gam)) + (-1 + 2*log(2) + log(gam)) / (gam^2 * 4)
        m1 = 1 / gam^2
        u = (1/2:N .- 1/2) * K / N
        dn = sech.(u) .+ 0.25 * m1 * (sinh.(u) .* cosh.(u) .+ u) .* tanh.(u) .* sech.(u)
    else
        kp = 1 - (1/gam)^2
        K = ellipticK(kp)  # Complete elliptic integral of the first kind
        dn = jellip.("dn",(1:2:(2*N - 1)) * K / (2*N); m=kp)  # Jacobian elliptic functions
    end

    # Optimal shift parameters Z(T([a, b]), T([c, d]))
    p1 = gam * dn

    # Solve for zeros and poles of rational function on [a, b] and [c, d]
    p = Tinv.(-p1)
    q = Tinv.(p1)

    return p, q
end

function mobiusT(II)
    # Given I = [a, b, c, d] where [a, b] and [c, d] are two disjoint intervals on the real line,
    # T(I) maps the points to [-γ, -1, 1, γ]. M is the cross-ratio. 
    a, b, c, d = II

    # Cross-ratio
    M = abs((c - a) * (d - b) / ((c - b) * abs(d - a)))
    gam = -1 + 2*M + 2*sqrt(M^2 - M)

    # Parameters for the Möbius transform
    A = -gam*a*(1 - gam) + gam*(c - gam*d) + c*gam - gam*d
    B = -gam*a*(c*gam - d) - a*(c*gam - gam*d) - gam*(c*d - gam*d*c)
    C = a*(1 - gam) + gam*(c - d) + c*gam - d
    D = -gam*a*(c - d) - a*(c - gam*d) + c*d - gam*d*c

    T = z -> (A*z + B) / (C*z + D)
    Tinv = z -> (D*z - B) / (-C*z + A)

    return T, Tinv, gam, M
end

function fadi(A, B, U, V, p=nothing, q=nothing; tol=eps())
    #= factored ADI:
    %
    % fadi(A, B, U, V, p, q)
    % solves AX - XB = U*V' in low rank form using factored ADI with 
    % ADI shift parameters provided by vectors p, q.
    % OUTPUT: ZZ*DD*YY' \approx =  X.
    %
    % fadi(A, B, U, V, p, q, tol) is as above, except that compression is
    % applied as the solution is computed. 
    % The compressed solution matches the uncompressed solution
    % to a relative accuracy of tol (wrt to the operator norm). 
    %
    % fadi(A, B, U, V) If A and B have real eigenvalues that are contained in 
    % disjoint intervals, the optimal shift parameters are computed automatically
    % and the problem is solved to a relative accuracy of approximately machine 
    % precision. 
    %
    % fadi(A, B, U, V; tol=Tol) is as above, except the relative accuracy of the 
    % of the solution is specified by Tol. 
    %
    % See getshifts_adi and getshifts_smith for help computing shift parameters. 
    %
    % References: 
    %
    % [1] Benner, Peter, Ren-Cang Li, and Ninoslav Truhar. 
    % "On the ADI method for Sylvester equations." J. of Comp. and App. Math.
    % 233.4 (2009): 1035-1045. 

    % code written by Heather Wilber (heatherw3521@gmail.com)
    % Jan. 2018
    %%=#
    m, r = size(U)
    n = size(V,1)
    compute_shifts = false
    
    #In = I(n)
    #Im = I(m)
    ZZ = []
    YY = []
    DD = []
    
    # Determine shift parameters if not provided
    if p === nothing || q === nothing
        compute_shifts = true
    end
    
    if compute_shifts
        # Find intervals where eigenvalues live
        a = eigvals(A)[1]
        b = eigvals(A)[end]
        c = eigvals(B)[1]
        d = eigvals(B)[end]

        # Ensure eigenvalues are real
        if any(abs.(imag.([a, b, c, d])) .> 1e-10)
            error("ADI:fadi:cannot automatically compute shift parameters unless the eigenvalues of A and B in AX - XB = F are contained in real, disjoint intervals.")
        end
        
        # Check for overlapping intervals
        I1 = (min(a, b), max(a, b))
        I2 = (min(c, d), max(c, d))
        if (I1[1] < I2[2] && I1[2] > I2[1]) || (I2[1] < I1[2] && I2[2] > I1[1])
            error("ADI:fadi:cannot automatically compute shift parameters unless the eigenvalues of A and B in AX - XB = F are contained in real, disjoint intervals.")
        end
        
        # Get shift parameters using the getshifts_adi function
        p, q = getshifts_adi([I1..., I2...], tol)
    end
    
    Ns = length(p)
    cp = conj(p)
    cq = conj(q)
    B = B'
    
    # Factored ADI
    Z = (A - I*q[1]) \ U
    Y = (B - cp[1]*I) \ V
    ZZ = Z
    YY = Y
    DD = (q[1] - p[1]) * ones(r)
    
    for i in 1:(Ns-1)
        Z = Z + (A - q[i+1]*I) \ ((q[i+1] - p[i]) * Z)
        Y = Y + (B - cp[i+1]*I) \ ((cp[i+1] - cq[i]) * Y)
        
        ZZ = hcat(ZZ, Z)
        YY = hcat(YY, Y)
        DD = vcat(DD, (q[i+1] - p[i+1]) * ones(r))
    end
    
    return ZZ, diagm(DD), YY
end
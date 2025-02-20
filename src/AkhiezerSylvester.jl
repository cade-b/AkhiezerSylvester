using RecurrenceCoefficients, LinearAlgebra

#functions to generate evenly-spaced points on the unit circle
mgrid = (n,L) -> -L .+ 2*L*(0:n-1)/n
zgrid = n -> exp.(1im*mgrid(n,pi))

#use this to get convergence rate
function golden_section(bands;tol=√eps())
    φ⁻¹ = (√5-1)/2
    a = bands[1,2]
    b = bands[2,1]
    while b - a > tol
        c = b - (b - a) * φ⁻¹
        d = a + (b - a) * φ⁻¹
        (~,~,~,gt) = get_n_coeffs_and_ints_akh(bands, 0, [c,d])
        gc = real(gt[1])
        gd = real(gt[2])
        if gc > gd
            b = d
        else  # f(c) > f(d) to find the maximum
            a = c
        end
    end
    (~,~,~,gt) = get_n_coeffs_and_ints_akh(bands, 0, (a+b)/2)
    return real(gt[1])
end

function COMPRESS(J::Array,K::Array,ϵ=1e-13)
    (QJ, R) = qr(J)
    (L, QK) = lq(K)
    (U,Σ,V) = svd(R*L)
    
    nsvals = sum(Σ.>ϵ*sum(Σ))
    #println("Old rank:",size(J,2)," New rank:",nsvals)
    
    J = (QJ*U*Diagonal(.√Σ))[:,1:nsvals]
    K = (Diagonal(.√Σ)*V'*QK)[1:nsvals,:]
    return J,K
end

function lowrank_block_svd(A::Matrix,U::Array,V::Array,B::Matrix,coeffs::Vector,avec::Vector,bvec::Vector,maxiter=100,eg0=1.; σtol=1e-13, get_resid=false, store_rank=false, tru_sol=nothing, use_weight_compress=true)
    if store_rank
        rankJK = zeros(maxiter)
        rankWZ = zeros(maxiter)
    end
    if get_resid
        errvec = zeros(maxiter)
    end

    for k = 0:maxiter-1
        if k == 0
            global CpA = V #Vpₖ(A)
            global pBC = U #pₖ(B)U
            global Jₖ = -U #Uₖ
            global Kₖ = V #Vₖ
        elseif k == 1
            CpA = (Aₖ₋₁*A-avec[1]*Aₖ₋₁)/bvec[1] #update Vpₖ(B) using three-term recurrence
            pBC = (B*Bₖ₋₁-avec[1]*Bₖ₋₁)/bvec[1] #update pₖ(D)U using three-term recurrence
            Jₖ = [Jₖ₋₁ (avec[1]+1)*U]/bvec[1] #update Uₖ according to block form of derived recurrence
            Kₖ = [Kₖ₋₁*A; V] #update Vₖ according to block form of derived recurrence

            #define variables to store  Vpₖ₋₂(B), etc.
            global Aₖ₋₂ = Aₖ₋₁
            global Bₖ₋₂ = Bₖ₋₁
            global Jₖ₋₂ = Jₖ₋₁
            global Kₖ₋₂ = Kₖ₋₁
        else
            CpA = (Aₖ₋₁*A-bvec[k-1]*Aₖ₋₂-avec[k]*Aₖ₋₁)/bvec[k] #update Vpₖ(B) using three-term recurrence
            pBC = (B*Bₖ₋₁-bvec[k-1]*Bₖ₋₂-avec[k]*Bₖ₋₁)/bvec[k] #update pₖ(D)U using three-term recurrence
            
            Jₖ = [Jₖ₋₁ Bₖ₋₁ -avec[k]*Jₖ₋₁ -bvec[k-1]*Jₖ₋₂]/bvec[k] #update Jₖ according to block form of derived recurrence
            Kₖ = [Kₖ₋₁*A; V; Kₖ₋₁; Kₖ₋₂] #update Kₖ according to block form of derived recurrence
            
            #store Vpₖ₋₂(B), etc.
            Aₖ₋₂ = Aₖ₋₁
            Bₖ₋₂ = Bₖ₋₁
            Jₖ₋₂ = Jₖ₋₁
            Kₖ₋₂ = Kₖ₋₁
        end
    
        if use_weight_compress
            (Jₖ,Kₖ) = COMPRESS(Jₖ,Kₖ,σtol*eg0^k/5) #weighted compression of Jₖ and Kₖ
        else
            (Jₖ,Kₖ) = COMPRESS(Jₖ,Kₖ,σtol) #unweighted compression of Jₖ and Kₖ
        end
        
        #define variables to store  Vpₖ₋₁(B), etc.
        global Aₖ₋₁ = CpA
        global Bₖ₋₁ = pBC
        global Jₖ₋₁ = Jₖ
        global Kₖ₋₁ = Kₖ
        
        if k == 0
            #store approximate solution in low rank form
            global Wₖ = coeffs[k+1]*[U Jₖ]/2
            global Zₖ = [CpA; Kₖ]
        else
        #assemble Xₖ+UVpₖ(B) in block form
            Wₖ = [Wₖ coeffs[k+1]*U/2 coeffs[k+1]*Jₖ/2]
            Zₖ = [Zₖ; CpA; Kₖ]
        end
        
        (Wₖ,Zₖ) = COMPRESS(Wₖ,Zₖ,σtol) #compress Wₖ and Zₖ

        if store_rank
            rankJK[k+1] = size(Jₖ,2)
            rankWZ[k+1] = size(Wₖ,2)
        end
        
        if get_resid
            if tru_sol !== nothing #Compute true error
                errvec[k+1] = opnorm(Wₖ*Zₖ-tru_sol)
            else #Compute residual
                X = Wₖ*Zₖ
                errvec[k+1] = opnorm(X*A-B*X-U*V)
            end
        end
    end

    if store_rank && get_resid
        return Wₖ,Zₖ,rankJK,rankWZ,errvec
    elseif store_rank
        return Wₖ,Zₖ,rankJK,rankWZ
    elseif get_resid
        return Wₖ,Zₖ,errvec
    else
        return Wₖ,Zₖ
    end
end

struct AkhParams
    α::Vector
    avec::Vector
    bvec::Vector
    conv_rate::Number
    maxiter::Int
end

function lowrank_block_svd(A::Matrix,U::Array,V::Array,B::Matrix,akhp::AkhParams; σtol=1e-13, get_resid=false, store_rank=false, tru_sol=nothing, use_weight_compress=true)
    return lowrank_block_svd(A,U,V,B,akhp.α,akhp.avec,akhp.bvec,akhp.maxiter,akhp.conv_rate;σtol=σtol,get_resid=get_resid,store_rank=store_rank,tru_sol=tru_sol, use_weight_compress=use_weight_compress)
end

function get_params(bands::Array{Float64,2}, A::Matrix, B::Matrix; circ_size=1.25, num_quad_pts=800, tol=1e-14, numiter=nothing, unbounded_op=false)
    n,m = size(A,1), size(B,1)
    gt = golden_section(bands)
    egt = exp(gt)

    if numiter === nothing
        numiter = ceil(-log(egt,maximum([tol/(5(m+n)*(1-1/egt)),eps()/5]))) |> Int
    end
    α = zeros(ComplexF64,numiter+1)

    if unbounded_op
        n = round(num_quad_pts/2) |> Int
        gd = JacobiMappedInterval(-1,1,0,0)
        y = gd.grid(n)
        a, b = OperatorApproximation.Jacobi_ab(0.0,0.0)
        w = OperatorApproximation.Gauss_quad(a,b,n-1)[2]
        z = tan.(pi*y/2)
        sgnpts = [-ones(n); ones(n)]
        (avec,bvec,ints) = get_n_coeffs_and_ints_akh(bands, numiter, im*z)
        for j = 1:n
            α -= 2im*π*sgnpts[j]*w[j]*(z[j]^2+1)*ints[:,j]
        end
        return AkhParams(α,avec,bvec,egt,numiter)
    end

    cc(j) = (bands[j,1]+bands[j,2])/2
    rr(j) = circ_size*(bands[j,2]-bands[j,1])/2

    ctrpts = [] #quadrature nodes
    ptsper = [0; 0]
    weights = [] #quadrature weights
    total_length = sum(bands[:,2]-bands[:,1])
    for j = 1:2
        ptsper[j] = round(num_quad_pts*(bands[j,2]-bands[j,1])/total_length)
        append!(ctrpts,rr(j)*zgrid(ptsper[j]).+cc(j) .|> Complex)
        append!(weights,2π*im*rr(j)*(zgrid(ptsper[j]))/ptsper[j] .|> Complex)
    end
    sgnpts = [-ones(ptsper[1]);ones(ptsper[2])]
    
    (avec,bvec,ints) = get_n_coeffs_and_ints_akh(bands, numiter, ctrpts)

    for j = 1:length(ctrpts)
        α -= sgnpts[j]*weights[j]*ints[:,j]
    end

    return AkhParams(α,avec,bvec,egt,numiter)
end
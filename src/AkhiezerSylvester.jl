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
    
    if nsvals > 0
        #J = (QJ*U*Diagonal(.√Σ))[:,1:nsvals]
        J = QJ*U[:,1:nsvals]*Diagonal(.√Σ[1:nsvals])
        #K = (Diagonal(.√Σ)*V'*QK)[1:nsvals,:]
        K = Diagonal(.√Σ[1:nsvals])*(V[:,1:nsvals])'*QK
    else
        J = zeros(size(J,1),1)
        K = zeros(1,size(K,2))
    end
    return J,K
end

function lowrank_block_svd(A::Matrix,U::Array,V::Array,B::Matrix,coeffs::Vector,avec::Vector,bvec::Vector,maxiter=100,eg0=1.; σtol=1e-14, get_resid=false, store_rank=false, tru_sol=nothing, use_weight_compress=true)
    if store_rank
        rankJK = zeros(Int,maxiter)
        rankWZ = zeros(Int,maxiter)
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
        #assemble Xₖ+UVpₖ(A) in block form
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

function lowrank_block_svd(A::Matrix,U::Array,V::Array,B::Matrix,akhp::AkhParams; σtol=1e-14, get_resid=false, store_rank=false, tru_sol=nothing, use_weight_compress=true)
    return lowrank_block_svd(A,U,V,B,akhp.α,akhp.avec,akhp.bvec,akhp.maxiter,1/akhp.conv_rate;σtol=σtol,get_resid=get_resid,store_rank=store_rank,tru_sol=tru_sol, use_weight_compress=use_weight_compress)
end

function get_params(bands::Array{Float64,2}, A::Matrix, B::Matrix; circ_size=1.25, num_quad_pts=800, tol=1e-14, numiter=nothing, unbounded_op=false, gt=nothing)
    if gt === nothing
        gt = golden_section(bands)
    end
    n,m = size(A,1), size(B,1)
    egt = exp(gt)

    if numiter === nothing
        numiter = ceil(-log(egt,maximum([tol*(1-1/egt)/(5(m+n)),eps()/5]))) |> Int
    end
    α = zeros(ComplexF64,numiter+1)

    if unbounded_op
        #Uses same number of quadrature points per interval for simplicity. Should be adjusted if one interval is much larger than the other.
        #n = round(num_quad_pts/2) |> Int
        gd = JacobiMappedInterval(-1,1,0,0)
        y = gd.grid(num_quad_pts)
        a, b = OperatorApproximation.Jacobi_ab(0.0,0.0)
        w = 2OperatorApproximation.Gauss_quad(a,b,num_quad_pts-1)[2]
        z = tan.(pi*y/2)
        #sgnpts = [-ones(n); ones(n)]
        (avec,bvec,ints) = get_n_coeffs_and_ints_akh(bands, numiter, -im*z)
        for j = 1:num_quad_pts
            α += im*π*w[j]*(z[j]^2+1)*ints[:,j]
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

    return AkhParams(α,avec,bvec,1/egt,numiter)
end

function sylv_operator_inv(A::Matrix, B::Matrix, C::Matrix, a::Float64, b::Float64; maxiter=100)
    α = (b+a)/2
    c = (b-a)/2
    P₀ = C
    S₀ = 1/(sqrt(a)*sqrt(b))
    global Sₖ₋₁ = S₀
    global Pₖ₋₁ = P₀
    global Xₖ = S₀*P₀
    for k = 1:maxiter-1
        if k == 1
            Pₖ = (Pₖ₋₁*A-B*Pₖ₋₁-α*Pₖ₋₁)/c
        else
            Pₖ = (2/c)*(Pₖ₋₁*A-B*Pₖ₋₁-α*Pₖ₋₁)-Pₖ₋₂
        end
        Sₖ = Sₖ₋₁*(-α/c+sqrt(1+α/c)*sqrt(α/c-1))
        Xₖ += 2Sₖ*Pₖ
        
        Sₖ₋₁ = Sₖ
        global Pₖ₋₂ = Pₖ₋₁
        Pₖ₋₁ = Pₖ
    end
    return Xₖ
end

function sylv_operator_inv_low_rank(A::Matrix,U::Array,V::Array,B::Matrix, a::Float64, b::Float64, maxiter=100, eg0=nothing; σtol=1e-14, get_resid=false, store_rank=false, tru_sol=nothing, use_weight_compress=true)
    if store_rank
        rankJK = zeros(Int,maxiter)
        rankWZ = zeros(Int,maxiter)
    end
    if get_resid
        errvec = zeros(maxiter)
    end

    α = (b+a)/2
    c = (b-a)/2
    S₀ = 1/(sqrt(a)*sqrt(b))
    ϱinv = -α/c+sqrt(1+α/c)*sqrt(α/c-1)
    #println(S₀)
    for k = 0:maxiter-1
        if k == 0
            global Jₖ = U #Uₖ
            global Kₖ = V #Vₖ
            global Sₖ = S₀
        elseif k == 1
            Jₖ = [Jₖ₋₁ -B*Jₖ₋₁ -α*Jₖ₋₁]/c #update Jₖ according to Chebyshev recurrence
            Kₖ = [Kₖ₋₁*A; Kₖ₋₁; Kₖ₋₁] #update Kₖ according to Chebyshev recurrence
            #U = UniformScaling(-α)
            #Jₖ = [Jₖ₋₁ (-B+U)*Jₖ₋₁]/c #update Jₖ according to Chebyshev recurrence
           # Kₖ = [Kₖ₋₁*A; Kₖ₋₁] #update Kₖ according to Chebyshev recurrence
            Sₖ = Sₖ₋₁*ϱinv
            global Jₖ₋₂ = Jₖ₋₁
            global Kₖ₋₂ = Kₖ₋₁
        else          
            Jₖ = [(2/c)*Jₖ₋₁ -(2/c)*B*Jₖ₋₁ -(2α/c)*Jₖ₋₁ -Jₖ₋₂] #update Jₖ according to Chebyshev recurrence
            Kₖ = [Kₖ₋₁*A; Kₖ₋₁; Kₖ₋₁; Kₖ₋₂] #update Kₖ according to Chebyshev recurrence
            #U = UniformScaling(-(2α/c))
            #Jₖ = [(2/c)*Jₖ₋₁ (-(2/c)*B +U)*Jₖ₋₁ -Jₖ₋₂] #update Jₖ according to Chebyshev recurrence
            #Kₖ = [Kₖ₋₁*A; Kₖ₋₁; Kₖ₋₂] #update Kₖ according to Chebyshev recurrence
            Sₖ = Sₖ₋₁*ϱinv
            Jₖ₋₂ = Jₖ₋₁
            Kₖ₋₂ = Kₖ₋₁
        end
    
        if use_weight_compress
            #println([abs(Sₖ);abs(ϱinv^k/5)])
            if eg0 === nothing
                (Jₖ,Kₖ) = COMPRESS(Jₖ,Kₖ,abs(σtol/5ϱinv^k)) #weighted compression of Jₖ and Kₖ
            else 
                (Jₖ,Kₖ) = COMPRESS(Jₖ,Kₖ,σtol*eg0^k/5) #weighted compression of Jₖ and Kₖ
            end
        else
            (Jₖ,Kₖ) = COMPRESS(Jₖ,Kₖ,σtol) #unweighted compression of Jₖ and Kₖ
        end

        global Jₖ₋₁ = Jₖ
        global Kₖ₋₁ = Kₖ
        global Sₖ₋₁ = Sₖ
        
        if k == 0
            #store approximate solution in low-rank form
            global Wₖ = Sₖ*Jₖ
            global Zₖ = Kₖ
        else
        #assemble Xₖ₋₁+SₖPₖ in block form
            Wₖ = [Wₖ 2Sₖ*Jₖ]
            Zₖ = [Zₖ; Kₖ]
        end
        
        (Wₖ,Zₖ) = COMPRESS(Wₖ,Zₖ,σtol) #compress Wₖ and Zₖ

        if store_rank
            rankJK[k+1] = size(Jₖ,2)
            rankWZ[k+1] = size(Wₖ,2)
        end
        
        if get_resid
            if tru_sol !== nothing #Compute true error
                errvec[k+1] = norm(Wₖ*Zₖ-tru_sol)
            else #Compute residual
                X = Wₖ*Zₖ
                errvec[k+1] = norm(X*A-B*X-U*V)
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

struct AkhParamsInv
    a::Float64
    b::Float64
    maxiter::Int
    conv_rate::Number
end

function get_params_inv(bands::Array{Float64,2}, A::Matrix, B::Matrix; tol=1e-14, numiter=nothing, ϱ=nothing)
    a = bands[2,1]-bands[1,2]
    b = bands[2,2]-bands[1,1]

    if ϱ === nothing
        α = (b+a)/2
        c = (b-a)/2
        ϱ = abs(1/(-α/c+sqrt(1+α/c)*sqrt(α/c-1)))
    end
    S₀ = 1/(sqrt(a)*sqrt(b))

    n,m = size(A,1), size(B,1)

    if numiter === nothing
        #println(-log(ϱ,tol*(1-1/ϱ)/(10m*n)))
        #println(-log(ϱ,eps()/(S₀*√2)))
        numiter = ceil(-log(ϱ,maximum([tol*(1-1/ϱ)/(20*(m+n)),eps()/(S₀*√2)]))) |> Int
    end
    return AkhParamsInv(a,b,numiter,1/ϱ)
end

function sylv_operator_inv_low_rank(A::Matrix,U::Array,V::Array,B::Matrix, akhp::AkhParamsInv; σtol=1e-14, get_resid=false, store_rank=false, tru_sol=nothing, use_weight_compress=true)
    return sylv_operator_inv_low_rank(A,U,V,B,akhp.a,akhp.b,akhp.maxiter,1/akhp.conv_rate; σtol=σtol,get_resid=get_resid,store_rank=store_rank,tru_sol=tru_sol, use_weight_compress=use_weight_compress)
end

function sylv_operator_inv_low_rank_more_ints(A::Matrix,U::Array,V::Array,B::Matrix, coeffs::Vector, avec::Vector, bvec::Vector, maxiter=100, eg0=1.; σtol=1e-14, get_resid=false, store_rank=false, tru_sol=nothing, use_weight_compress=true)
    if store_rank
        rankJK = zeros(maxiter)
        rankWZ = zeros(maxiter)
    end
    if get_resid
        errvec = zeros(maxiter)
    end

    #println(S₀)
    for k = 0:maxiter-1
        if k == 0
            global Jₖ = U #Uₖ
            global Kₖ = V #Vₖ
        elseif k == 1
            Jₖ = [Jₖ₋₁ -B*Jₖ₋₁ -avec[1]*Jₖ₋₁]/bvec[1] #update Jₖ according to Chebyshev recurrence
            Kₖ = [Kₖ₋₁*A; Kₖ₋₁; Kₖ₋₁] #update Kₖ according to Chebyshev recurrence
        else          
            Jₖ = [Jₖ₋₁ -B*Jₖ₋₁ -avec[k]*Jₖ₋₁ -bvec[k-1]*Jₖ₋₂]/bvec[k] #update Jₖ according to Chebyshev recurrence
            Kₖ = [Kₖ₋₁*A; Kₖ₋₁; Kₖ₋₁; Kₖ₋₂] #update Kₖ according to Chebyshev recurrence
        end
    
        if use_weight_compress
            (Jₖ,Kₖ) = COMPRESS(Jₖ,Kₖ,σtol*eg0^k/5) #weighted compression of Jₖ and Kₖ
        else
            (Jₖ,Kₖ) = COMPRESS(Jₖ,Kₖ,σtol) #unweighted compression of Jₖ and Kₖ
        end
        
        global Jₖ₋₂ = Jₖ₋₁
        global Kₖ₋₂ = Kₖ₋₁
        global Jₖ₋₁ = Jₖ
        global Kₖ₋₁ = Kₖ
        
        if k == 0
            #store approximate solution in low-rank form
            global Wₖ = coeffs[1]*Jₖ
            global Zₖ = Kₖ
        else
        #assemble Xₖ₋₁+SₖPₖ in block form
            Wₖ = [Wₖ coeffs[k+1]*Jₖ]
            Zₖ = [Zₖ; Kₖ]
        end
        
        (Wₖ,Zₖ) = COMPRESS(Wₖ,Zₖ,σtol) #compress Wₖ and Zₖ

        if store_rank
            rankJK[k+1] = size(Jₖ,2)
            rankWZ[k+1] = size(Wₖ,2)
        end
        
        if get_resid
            if tru_sol !== nothing #Compute true error
                errvec[k+1] = norm(Wₖ*Zₖ-tru_sol)
            else #Compute residual
                X = Wₖ*Zₖ
                errvec[k+1] = norm(X*A-B*X-U*V)
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
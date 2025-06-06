using LinearAlgebra

function algorithm5(A, B, C₁, C₂; tol=1e-11, maxiter=100, get_resid=true)
    # orthogonalize columns of C₁
    V₁ = qr(C₁).Q
    vₖ = Matrix(V₁)

    # orthogonalize columns of C₂
    W₁ = qr(C₂).Q
    wₖ = Matrix(W₁)

    Vₖ = vₖ
    Wₖ = wₖ

    if get_resid
        resids = zeros(maxiter)
    end

    for k in 1:maxiter-1
        Aₖ = Vₖ' * A * Vₖ
        Bₖ = Wₖ' * B * Wₖ
        Cₖ = Vₖ' * C₁ * C₂' * Wₖ

        # Solve AₖY + YBₖ = Cₖ
        Yₖ = sylvester(Aₖ, Bₖ, -Cₖ)

        # compute residuals (expensively) if requested
        if get_resid
            Xₖ = Vₖ * Yₖ * Wₖ'
            resids[k] = norm(A*Xₖ + Xₖ*B - C₁*C₂')
            println("Iteration $k, residual = ", resids[k])
            if resids[k] < tol
                return Vₖ, Yₖ, Wₖ, resids[1:k]
            end
        end

        #Next blocks in Krylov space
        v̂ = A * vₖ
        ŵ = B' * wₖ

        #=Qv = qr(v̂).Q
        Qw = qr(ŵ).Q

        v̂ = Matrix(Qv)
        ŵ = Matrix(Qw)=#

        # Orthogonalize v̂ wrt Vₖ
        for j in 1:size(Vₖ, 2)
            v̂ -= Vₖ[:, j] * (Vₖ[:, j]' * v̂)
        end

        # reorthgonalize
        for j in 1:size(Vₖ, 2)
            v̂ -= Vₖ[:, j] * (Vₖ[:, j]' * v̂)
        end

        # Orthogonalize ŵ wrt Wₖ
        for j in 1:size(Wₖ, 2)
            ŵ -= Wₖ[:, j] * (Wₖ[:, j]' * ŵ)
        end

        # reorthgonalize
        for j in 1:size(Wₖ, 2)
            ŵ -= Wₖ[:, j] * (Wₖ[:, j]' * ŵ)
        end

        # Orthogonalize within block
        Qv = qr(v̂).Q
        Qw = qr(ŵ).Q

        v̂ = Matrix(Qv)
        ŵ = Matrix(Qw)

        # update
        Vₖ = hcat(Vₖ, v̂)
        Wₖ = hcat(Wₖ, ŵ)

        vₖ = v̂
        wₖ = ŵ
    end
    # final Sylvester solve if max iterations hit
    Aₖ = Vₖ' * A * Vₖ
    Bₖ = Wₖ' * B * Wₖ
    Cₖ = Vₖ' * C₁ * C₂' * Wₖ

    # Solve Aₖ*Y + Y*Bₖ = Cₖ
    Yₖ = sylvester(Aₖ, Bₖ, -Cₖ)
    if get_resid
        Xₖ = Vₖ * Yₖ * Wₖ'
        resids[maxiter] = norm(A*Xₖ + Xₖ*B - C₁*C₂')
        println("Iteration $maxiter, residual = ", resids[maxiter])
        return Vₖ, Yₖ, Wₖ, resids
    else
        return Vₖ, Yₖ, Wₖ
    end
end

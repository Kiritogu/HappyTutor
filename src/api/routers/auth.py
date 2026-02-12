from __future__ import annotations

from fastapi import APIRouter, Depends, Header, HTTPException, Request, status

from src.api.dependencies.auth import get_auth_service, get_current_user
from src.services.auth.auth_service import AuthService
from src.services.auth.schemas import LoginRequest, LogoutRequest, RefreshRequest, RegisterRequest

router = APIRouter()


@router.post("/register")
async def register(
    payload: RegisterRequest,
    auth_service: AuthService = Depends(get_auth_service),
):
    try:
        return auth_service.register(email=payload.email, password=payload.password)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc


@router.post("/login")
async def login(
    payload: LoginRequest,
    request: Request,
    auth_service: AuthService = Depends(get_auth_service),
    user_agent: str | None = Header(default=None),
):
    try:
        return auth_service.login(
            email=payload.email,
            password=payload.password,
            created_ip=request.client.host if request.client else None,
            user_agent=user_agent,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(exc)) from exc


@router.post("/refresh")
async def refresh(
    payload: RefreshRequest,
    request: Request,
    auth_service: AuthService = Depends(get_auth_service),
    user_agent: str | None = Header(default=None),
):
    try:
        return auth_service.refresh_tokens(
            payload.refresh_token,
            created_ip=request.client.host if request.client else None,
            user_agent=user_agent,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(exc)) from exc


@router.post("/logout")
async def logout(
    payload: LogoutRequest,
    auth_service: AuthService = Depends(get_auth_service),
):
    try:
        success = auth_service.logout(payload.refresh_token)
        return {"success": bool(success)}
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(exc)) from exc


@router.get("/me")
async def me(
    authorization: str | None = Header(default=None),
    auth_service: AuthService = Depends(get_auth_service),
):
    user = get_current_user(auth_service=auth_service, authorization=authorization)
    return {"user": user}


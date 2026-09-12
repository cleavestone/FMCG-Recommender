import jwt
from fastapi import APIRouter, Depends, Header, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlmodel import Session

from fmcg_reco.config import settings
from fmcg_reco.serving.auth.dependencies import get_current_user
from fmcg_reco.serving.auth.models import User
from fmcg_reco.serving.auth.schemas import (
    AccessToken,
    RefreshRequest,
    RegisterRequest,
    Token,
    UserRead,
)
from fmcg_reco.serving.auth.security import create_access_token, create_refresh_token, decode_token
from fmcg_reco.serving.auth.store import authenticate_user, create_user, get_session, get_user

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/token", response_model=Token, summary="Log in")
def login(
    form_data: OAuth2PasswordRequestForm = Depends(), session: Session = Depends(get_session)
) -> Token:
    user = authenticate_user(session, form_data.username, form_data.password)
    if not user:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "incorrect username or password")
    return Token(
        access_token=create_access_token(user.username),
        refresh_token=create_refresh_token(user.username),
    )


@router.post("/refresh", response_model=AccessToken, summary="Refresh an access token")
def refresh(body: RefreshRequest, session: Session = Depends(get_session)) -> AccessToken:
    invalid = HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid refresh token")
    try:
        payload = decode_token(body.refresh_token)
    except jwt.PyJWTError as exc:
        raise invalid from exc
    if payload.get("type") != "refresh":
        raise invalid
    username = payload.get("sub", "")
    if not get_user(session, username):
        raise invalid
    return AccessToken(access_token=create_access_token(username))


@router.get("/me", response_model=UserRead, summary="Current user")
def me(current_user: User = Depends(get_current_user)) -> UserRead:
    return UserRead(id=current_user.id, username=current_user.username)


@router.post(
    "/register", response_model=UserRead, status_code=status.HTTP_201_CREATED, summary="Create a user (admin-gated)"
)
def register(
    body: RegisterRequest,
    session: Session = Depends(get_session),
    x_admin_key: str = Header(...),
) -> UserRead:
    if x_admin_key != settings.admin_api_key:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "invalid admin key")
    if get_user(session, body.username):
        raise HTTPException(status.HTTP_409_CONFLICT, "username already exists")
    user = create_user(session, body.username, body.password)
    return UserRead(id=user.id, username=user.username)

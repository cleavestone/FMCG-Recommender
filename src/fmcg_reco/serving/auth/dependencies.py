import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlmodel import Session

from fmcg_reco.serving.auth.models import User
from fmcg_reco.serving.auth.security import decode_token
from fmcg_reco.serving.auth.store import get_session, get_user

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/token")


def get_current_user(
    token: str = Depends(oauth2_scheme), session: Session = Depends(get_session)
) -> User:
    unauthorized = HTTPException(
        status.HTTP_401_UNAUTHORIZED,
        "could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = decode_token(token)
    except jwt.PyJWTError as exc:
        raise unauthorized from exc
    if payload.get("type") != "access":
        raise unauthorized
    user = get_user(session, payload.get("sub", ""))
    if not user:
        raise unauthorized
    return user

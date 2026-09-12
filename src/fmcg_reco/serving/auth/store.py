from collections.abc import Iterator

from sqlmodel import Session, SQLModel, create_engine, select

from fmcg_reco.config import settings
from fmcg_reco.serving.auth.models import User
from fmcg_reco.serving.auth.security import hash_password, verify_password

engine = create_engine(settings.database_url, connect_args={"check_same_thread": False})


def init_db() -> None:
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        if not session.exec(select(User)).first():
            session.add(User(username="demo", hashed_password=hash_password("demo-password")))
            session.commit()


def get_session() -> Iterator[Session]:
    with Session(engine) as session:
        yield session


def get_user(session: Session, username: str) -> User | None:
    return session.exec(select(User).where(User.username == username)).first()


def authenticate_user(session: Session, username: str, password: str) -> User | None:
    user = get_user(session, username)
    if user and verify_password(password, user.hashed_password):
        return user
    return None


def create_user(session: Session, username: str, password: str) -> User:
    user = User(username=username, hashed_password=hash_password(password))
    session.add(user)
    session.commit()
    session.refresh(user)
    return user

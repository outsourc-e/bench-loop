import pytest
from src import ConnectionPool


def test_basic_acquire_release():
    pool = ConnectionPool(3)
    conn = pool.acquire()
    assert pool.active_count == 1
    assert pool.available_count == 2
    pool.release(conn)
    assert pool.active_count == 0
    assert pool.available_count == 3


def test_exhaust_pool():
    pool = ConnectionPool(2)
    pool.acquire()
    pool.acquire()
    with pytest.raises(RuntimeError, match="No connections"):
        pool.acquire()


def test_release_unknown():
    pool = ConnectionPool(3)
    with pytest.raises(ValueError):
        pool.release(99)


def test_close_clears_everything():
    """After close, both active and available should be 0."""
    pool = ConnectionPool(3)
    pool.acquire()
    pool.acquire()
    assert pool.active_count == 2
    pool.close()
    assert pool.is_closed
    assert pool.active_count == 0
    assert pool.available_count == 0


def test_acquire_after_close():
    pool = ConnectionPool(3)
    pool.close()
    with pytest.raises(RuntimeError, match="closed"):
        pool.acquire()


def test_fifo_order():
    """Connections should be returned in FIFO order."""
    pool = ConnectionPool(3)
    c0 = pool.acquire()
    c1 = pool.acquire()
    pool.release(c0)
    pool.release(c1)
    assert pool.acquire() == c0
    assert pool.acquire() == c1

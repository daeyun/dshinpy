import pytest
import peewee
from playhouse.sqlite_ext import SqliteExtDatabase
from dshin import data

db = SqliteExtDatabase(':memory:', threadlocals=True)


class BaseModel(peewee.Model):
    class Meta:
        database = db


class User(BaseModel):
    username = peewee.CharField(unique=True, max_length=255, index=True)
    filename = peewee.TextField()


@pytest.fixture('function')
def query():
    return User.select().dicts()


def setup_module(module):
    module.db.create_tables([User])
    User.create(username='first', filename='sample.npz')
    User.create(username='second', filename='sample2.npz')


def teardown_module(module):
    module.db.close()


def test_paginator_next_zero(query):
    paginator = data.QueryPaginator(query)

    result = paginator.next(0)
    assert len(result) == 0

    result = paginator.next(1)
    assert len(result) == 1
    assert result[0]['username'] == 'first'


def test_paginator_next_one(query):
    paginator = data.QueryPaginator(query)

    result = paginator.next(1)
    assert len(result) == 1
    assert result[0]['username'] == 'first'

    result = paginator.next(1)
    assert len(result) == 1
    assert result[0]['username'] == 'second'

    result = paginator.next(1)
    assert len(result) == 0

    result = paginator.next(1)
    assert len(result) == 1
    assert result[0]['username'] == 'first'


def test_paginator_next_two(query):
    paginator = data.QueryPaginator(query)

    result = paginator.next(2)
    assert len(result) == 2
    assert result[0]['username'] == 'first'
    assert result[1]['username'] == 'second'

    result = paginator.next(2)
    assert len(result) == 0

    result = paginator.next(2)
    assert len(result) == 2
    assert result[0]['username'] == 'first'
    assert result[1]['username'] == 'second'


def test_paginator_next_three(query):
    paginator = data.QueryPaginator(query)

    result = paginator.next(3)
    assert len(result) == 2
    assert result[0]['username'] == 'first'
    assert result[1]['username'] == 'second'

    result = paginator.next(3)
    assert len(result) == 0

    result = paginator.next(3)
    assert len(result) == 2
    assert result[0]['username'] == 'first'
    assert result[1]['username'] == 'second'


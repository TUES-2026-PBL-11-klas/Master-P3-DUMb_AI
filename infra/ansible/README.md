# MongoDB Replica Set With Ansible

This directory prepares the MongoDB replication work before the real server,
Raspberry Pi, and MacBook access is available.

The goal is to make the setup reproducible:

1. configure MongoDB on every host,
2. enable the same replica set name everywhere,
3. start/restart MongoDB,
4. initiate the replica set once from the primary candidate,
5. verify that a primary is elected.

## Files

- `inventory.example.ini` - placeholder host list. Copy it before adding real IPs.
- `group_vars/mongo.yml` - shared replica set settings.
- `templates/mongod.conf.j2` - MongoDB config template.
- `mongo_replica.yml` - Ansible playbook for config + replica set initiation.

## Before Running

Install Ansible on the control machine:

```bash
python3 -m pip install ansible
```

Copy the example inventory:

```bash
cp infra/ansible/inventory.example.ini infra/ansible/inventory.ini
```

Edit `infra/ansible/inventory.ini` with the real hosts:

```ini
[mongo]
mongo1 ansible_host=<linux-server-ip> ansible_user=<ssh-user> mongo_replica_priority=2
mongo2 ansible_host=<raspberry-pi-ip> ansible_user=<ssh-user> mongo_replica_priority=1
mongo3 ansible_host=<macbook-ip> ansible_user=<ssh-user> mongo_replica_priority=1

[mongo_primary]
mongo1
```

Check that SSH works:

```bash
ansible -i infra/ansible/inventory.ini mongo -m ping
```

## Important Assumptions

- MongoDB is available as `mongod` on every host.
- `mongosh` is available on the primary host.
- The hosts can reach each other on `27017`.
- The replica set name is `rs0`.
- The setup uses an odd number of MongoDB nodes, normally 3.

For the first MVP, package installation is intentionally disabled in
`group_vars/mongo.yml`:

```yaml
mongo_install_packages: false
```

This is safer until the exact Linux distribution, Raspberry Pi architecture,
and MacBook Homebrew setup are known. After that, installation tasks can be
added per OS.

## Run The Playbook

```bash
ansible-playbook -i infra/ansible/inventory.ini infra/ansible/mongo_replica.yml
```

## Verify Manually

SSH into the primary host and run:

```bash
mongosh --host localhost --port 27017
```

Then:

```javascript
rs.status()
```

Expected result:

- one member is `PRIMARY`,
- the other members are `SECONDARY`,
- the replica set name is `rs0`.

## Initialize DUMb_AI Collections

After the replica set is healthy, run the normal Mongo initialization script
against the primary:

```bash
mongosh "mongodb://<primary-host>:27017/?replicaSet=rs0" infra/mongo/init_db.js
```

This creates:

- collections,
- normal indexes,
- `schema_versions`,
- vector search index configuration, if supported by the MongoDB runtime.

## Notes About Vector Search

MongoDB replication and MongoDB Vector Search are related but separate concerns.
The replica set copies stored data between nodes, but the team still needs to
verify the exact local MongoDB runtime used for Vector Search.

Questions to confirm before production-like use:

- Are we using regular MongoDB Community, MongoDB Enterprise, Atlas Local, or
  another local container/runtime?
- Does that runtime support Vector Search in the same deployment mode?
- Are vector indexes created on all replica members or only queryable through
  the primary?
- Will the app connect using `replicaSet=rs0`?

For the MVP, it is acceptable to use the replica set for normal MongoDB
resilience and keep Vector Search testing focused on the supported local
runtime until this is confirmed.


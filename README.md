# lstm-examples

This code depends on [Ocropy][1] and the Scientific Python stack.

To set it up, run:

```
git clone https://github.com/danvk/lstm-examples.git
cd lstm-examples
git clone https://github.com/tmbdev/ocropy/
cp -r ocropy/ocrolib .
rm -rf ocropy
```

To try the [Reber Grammar][2] example, you should just be able to run:

```
./reber.py
```

[1]: https://github.com/tmbdev/ocropy/
[2]: http://www.willamette.edu/~gorr/classes/cs449/reber.html

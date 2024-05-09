local factory = require 'factories.ctf_large_factory'

return factory.createLevelApi{
	mapName = 'ctf_large',
    episodeLengthSeconds = 1,
    botCount = 3
}
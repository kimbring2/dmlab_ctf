local factory = require 'factories.ctf_simple_factory'

return factory.createLevelApi{
	mapName = 'ctf_simple',
    episodeLengthSeconds = 1,
    botCount = 1
}